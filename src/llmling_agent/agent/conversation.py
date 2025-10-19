"""Conversation management for LLMling agent."""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self, overload
from uuid import UUID, uuid4

from llmling import BasePrompt, PromptMessage, StaticPrompt
from llmling.config.models import BaseResource
from psygnal import Signal
from upathtools import read_path

from llmling_agent.log import get_logger
from llmling_agent.messaging.message_container import ChatMessageContainer
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.utils.count_tokens import count_tokens
from llmling_agent.utils.now import get_now
from llmling_agent_config.session import SessionQuery


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Sequence
    from datetime import datetime
    from types import TracebackType

    from llmling.config.models import Resource
    from llmling.prompts import PromptType
    from toprompt import AnyPromptType
    from upath.types import JoinablePathLike

    from llmling_agent.agent.agent import Agent
    from llmling_agent.common_types import MessageRole, SessionIdType
    from llmling_agent_config.session import MemoryConfig

logger = get_logger(__name__)

OverrideMode = Literal["replace", "append"]
type PromptInput = str | BasePrompt


def _to_base_prompt(prompt: PromptInput) -> BasePrompt:
    """Convert input to BasePrompt instance."""
    if isinstance(prompt, str):
        msg = PromptMessage(role="system", content=prompt)
        return StaticPrompt(
            name="System prompt", description="System prompt", messages=[msg]
        )
    return prompt


class ConversationManager:
    """Manages conversation state and system prompts."""

    @dataclass(frozen=True)
    class HistoryCleared:
        """Emitted when chat history is cleared."""

        session_id: str
        timestamp: datetime = field(default_factory=get_now)

    history_cleared = Signal(HistoryCleared)

    def __init__(
        self,
        agent: Agent[Any],
        session_config: MemoryConfig | None = None,
        *,
        resources: Sequence[Resource | PromptType | str] = (),
    ):
        """Initialize conversation manager.

        Args:
            agent: instance to manage
            session_config: Optional MemoryConfig
            resources: Optional paths to load as context
        """
        self._agent = agent
        self.chat_messages = ChatMessageContainer()
        self._last_messages: list[ChatMessage] = []
        self._pending_messages: deque[ChatMessage] = deque()
        self._config = session_config
        self._resources = list(resources)  # Store for async loading
        # Generate new ID if none provided
        self.id = str(uuid4())

        if session_config is not None and session_config.session is not None:
            storage = self._agent.context.storage
            self._current_history = storage.filter_messages_sync(session_config.session)
            if session_config.session.name:
                self.id = session_config.session.name

        # Note: max_messages and max_tokens will be handled in add_message/get_history
        # to maintain the rolling window during conversation

    def get_initialization_tasks(self) -> list[Coroutine[Any, Any, Any]]:
        """Get all initialization coroutines."""
        self._resources = []  # Clear so we dont load again on async init
        return [self.load_context_source(source) for source in self._resources]

    async def __aenter__(self) -> Self:
        """Initialize when used standalone."""
        if tasks := self.get_initialization_tasks():
            await asyncio.gather(*tasks)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Clean up any pending messages."""
        self._pending_messages.clear()

    def __bool__(self) -> bool:
        return bool(self._pending_messages) or bool(self.chat_messages)

    def __repr__(self) -> str:
        return f"ConversationManager(id={self.id!r})"

    def __prompt__(self) -> str:
        if not self.chat_messages:
            return "No conversation history"

        last_msgs = self.chat_messages[-2:]
        parts = ["Recent conversation:"]
        parts.extend(msg.format() for msg in last_msgs)
        return "\n".join(parts)

    @overload
    def __getitem__(self, key: int) -> ChatMessage[Any]: ...

    @overload
    def __getitem__(self, key: slice | str) -> list[ChatMessage[Any]]: ...

    def __getitem__(
        self, key: int | slice | str
    ) -> ChatMessage[Any] | list[ChatMessage[Any]]:
        """Access conversation history.

        Args:
            key: Either:
                - Integer index for single message
                - Slice for message range
                - Agent name for conversation history with that agent
        """
        match key:
            case int():
                return self.chat_messages[key]
            case slice():
                return list(self.chat_messages[key])
            case str():
                query = SessionQuery(name=key)
                return self._agent.context.storage.filter_messages_sync(query=query)
            case _:
                msg = f"Invalid key type: {type(key)}"
                raise TypeError(msg)

    def __contains__(self, item: Any) -> bool:
        """Check if item is in history."""
        return item in self.chat_messages

    def __len__(self) -> int:
        """Get length of history."""
        return len(self.chat_messages)

    def get_message_tokens(self, message: ChatMessage) -> int:
        """Get token count for a single message."""
        content = "\n".join(message.format())
        return count_tokens(content, self._agent.model_name)

    async def format_history(
        self,
        *,
        max_tokens: int | None = None,
        include_system: bool = False,
        format_template: str | None = None,
        num_messages: int | None = None,  # Add this parameter
    ) -> str:
        """Format conversation history as a single context message.

        Args:
            max_tokens: Optional limit to include only last N tokens
            include_system: Whether to include system messages
            format_template: Optional custom format (defaults to agent/message pairs)
            num_messages: Optional limit to include only last N messages
        """
        template = format_template or "Agent {agent}: {content}\n"
        messages: list[str] = []
        token_count = 0

        # Get messages, optionally limited
        history: Sequence[ChatMessage[Any]] = self.chat_messages
        if num_messages:
            history = history[-num_messages:]

        if max_tokens:
            history = list(reversed(history))  # Start from newest when token limited

        for msg in history:
            # Check role directly from ChatMessage
            if not include_system and msg.role == "system":
                continue
            name = msg.name or msg.role.title()
            formatted = template.format(agent=name, content=str(msg.content))

            if max_tokens:
                # Count tokens in this message
                if msg.cost_info:
                    msg_tokens = msg.cost_info.token_usage["total"]
                else:
                    # Fallback to tiktoken if no cost info
                    msg_tokens = self.get_message_tokens(msg)

                if token_count + msg_tokens > max_tokens:
                    break
                token_count += msg_tokens
                # Add to front since we're going backwards
                messages.insert(0, formatted)
            else:
                messages.append(formatted)

        return "\n".join(messages)

    async def load_context_source(self, source: Resource | PromptType | str):
        """Load context from a single source."""
        try:
            match source:
                case str():
                    await self.add_context_from_path(source)
                case BaseResource():
                    await self.add_context_from_resource(source)
                case BasePrompt():
                    await self.add_context_from_prompt(source)
        except Exception:
            msg = "Failed to load context from %s"
            logger.exception(msg, "file" if isinstance(source, str) else source.type)

    def load_history_from_database(
        self,
        session: SessionIdType | SessionQuery = None,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        roles: set[MessageRole] | None = None,
        limit: int | None = None,
    ):
        """Load conversation history from database.

        Args:
            session: Session ID or query config
            since: Only include messages after this time (override)
            until: Only include messages before this time (override)
            roles: Only include messages with these roles (override)
            limit: Maximum number of messages to return (override)
        """
        storage = self._agent.context.storage
        match session:
            case SessionQuery() as query:
                # Override query params if provided
                if since is not None or until is not None or roles or limit:
                    update = {
                        "since": since.isoformat() if since else None,
                        "until": until.isoformat() if until else None,
                        "roles": roles,
                        "limit": limit,
                    }
                    query = query.model_copy(update=update)
                if query.name:
                    self.id = query.name
            case str() | UUID():
                self.id = str(session)
                query = SessionQuery(
                    name=self.id,
                    since=since.isoformat() if since else None,
                    until=until.isoformat() if until else None,
                    roles=roles,
                    limit=limit,
                )
            case None:
                # Use current session ID
                query = SessionQuery(
                    name=self.id,
                    since=since.isoformat() if since else None,
                    until=until.isoformat() if until else None,
                    roles=roles,
                    limit=limit,
                )
            case _:
                msg = f"Invalid type for session: {type(session)}"
                raise ValueError(msg)
        self.chat_messages.clear()
        self.chat_messages.extend(storage.filter_messages_sync(query))

    def get_history(
        self,
        include_pending: bool = True,
        do_filter: bool = True,
    ) -> list[ChatMessage]:
        """Get conversation history.

        Args:
            include_pending: Whether to include pending messages
            do_filter: Whether to apply memory config limits (max_tokens, max_messages)

        Returns:
            Filtered list of messages in chronological order
        """
        if include_pending and self._pending_messages:
            self.chat_messages.extend(self._pending_messages)
            self._pending_messages.clear()

        # 2. Start with original history
        history: Sequence[ChatMessage[Any]] = self.chat_messages

        # 3. Only filter if needed
        if do_filter and self._config:
            # First filter by message count (simple slice)
            if self._config.max_messages:
                history = history[-self._config.max_messages :]

            # Then filter by tokens if needed
            if self._config.max_tokens:
                token_count = 0
                filtered = []
                # Collect messages from newest to oldest until we hit the limit
                for msg in reversed(history):
                    msg_tokens = self.get_message_tokens(msg)
                    if token_count + msg_tokens > self._config.max_tokens:
                        break
                    token_count += msg_tokens
                    filtered.append(msg)
                history = list(reversed(filtered))

        return list(history)

    def get_pending_messages(self) -> list[ChatMessage]:
        """Get messages that will be included in next interaction."""
        return list(self._pending_messages)

    def clear_pending(self):
        """Clear pending messages without adding them to history."""
        self._pending_messages.clear()

    def set_history(self, history: list[ChatMessage]):
        """Update conversation history after run."""
        self.chat_messages.clear()
        self.chat_messages.extend(history)

    def clear(self):
        """Clear conversation history and prompts."""
        self.chat_messages = ChatMessageContainer()
        self._last_messages = []
        event = self.HistoryCleared(session_id=str(self.id))
        self.history_cleared.emit(event)

    @asynccontextmanager
    async def temporary_state(
        self,
        history: list[AnyPromptType] | SessionQuery | None = None,
        *,
        replace_history: bool = False,
    ) -> AsyncIterator[Self]:
        """Temporarily set conversation history.

        Args:
            history: Optional list of prompts to use as temporary history.
                    Can be strings, BasePrompts, or other prompt types.
            replace_history: If True, only use provided history. If False, append
                    to existing history.
        """
        from toprompt import to_prompt

        old_history = self.chat_messages.copy()

        try:
            messages: Sequence[ChatMessage[Any]] = ChatMessageContainer()
            if history is not None:
                if isinstance(history, SessionQuery):
                    messages = await self._agent.context.storage.filter_messages(history)
                else:
                    messages = [
                        ChatMessage(content=await to_prompt(p), role="user")
                        for p in history
                    ]

            if replace_history:
                self.chat_messages = ChatMessageContainer(messages)
            else:
                self.chat_messages.extend(messages)

            yield self

        finally:
            self.chat_messages = old_history

    def add_chat_messages(self, messages: Sequence[ChatMessage]):
        """Add new messages to history and update last_messages."""
        self._last_messages = list(messages)
        self.chat_messages.extend(messages)

    @property
    def last_run_messages(self) -> list[ChatMessage]:
        """Get messages from the last run converted to our format."""
        return self._last_messages

    def add_context_message(
        self,
        content: str,
        source: str | None = None,
        **metadata: Any,
    ):
        """Add a context message.

        Args:
            content: Text content to add
            source: Description of content source
            **metadata: Additional metadata to include with the message
        """
        meta_str = ""
        if metadata:
            meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            meta_str = f"\nMetadata:\n{meta_str}\n"

        header = f"Content from {source}:" if source else "Additional context:"
        formatted = f"{header}{meta_str}\n{content}\n"

        chat_message = ChatMessage(
            content=formatted,
            role="user",
            name="user",
            model=self._agent.model_name,
            metadata=metadata,
            conversation_id="context",  # TODO: should probably allow DB field to be NULL
        )
        self._pending_messages.append(chat_message)
        # Emit as user message - will trigger logging through existing flow
        self._agent.message_received.emit(chat_message)

    async def add_context_from_path(
        self,
        path: JoinablePathLike,
        *,
        convert_to_md: bool = False,
        **metadata: Any,
    ):
        """Add file or URL content as context message.

        Args:
            path: Any UPath-supported path
            convert_to_md: Whether to convert content to markdown
            **metadata: Additional metadata to include with the message

        Raises:
            ValueError: If content cannot be loaded or converted
        """
        from upathtools import to_upath

        path_obj = to_upath(path)
        if convert_to_md:
            content = await self._agent.context.converter.convert_file(path)
            source = f"markdown:{path_obj.name}"
        else:
            content = await read_path(path)
            source = f"{path_obj.protocol}:{path_obj.name}"
        self.add_context_message(content, source=source, **metadata)

    async def add_context_from_resource(self, resource: Resource | str):
        """Add content from a LLMling resource."""
        if not self._agent.runtime:
            msg = "No runtime available"
            raise RuntimeError(msg)

        if isinstance(resource, str):
            content = await self._agent.runtime.load_resource(resource)
            self.add_context_message(
                str(content.content),
                source=f"Resource {resource}",
                mime_type=content.metadata.mime_type,
                **content.metadata.extra,
            )
        else:
            loader = self._agent.runtime._loader_registry.get_loader(resource)
            async for content in loader.load(resource):
                self.add_context_message(
                    str(content.content),
                    source=f"{resource.type}:{resource.uri}",
                    mime_type=content.metadata.mime_type,
                    **content.metadata.extra,
                )

    async def add_context_from_prompt(
        self,
        prompt: PromptType,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Add rendered prompt content as context message.

        Args:
            prompt: LLMling prompt (static, dynamic, or file-based)
            metadata: Additional metadata to include with the message
            kwargs: Optional kwargs for prompt formatting
        """
        try:
            # Format the prompt using LLMling's prompt system
            messages = await prompt.format(kwargs)
            # Extract text content from all messages
            content = "\n\n".join(msg.get_text_content() for msg in messages)

            self.add_context_message(
                content,
                source=f"prompt:{prompt.name or prompt.type}",
                prompt_args=kwargs,
                **(metadata or {}),
            )
        except Exception as e:
            msg = f"Failed to format prompt: {e}"
            raise ValueError(msg) from e

    def get_history_tokens(self) -> int:
        """Get token count for current history."""
        # Use cost_info if available
        return self.chat_messages.get_history_tokens(self._agent.model_name)

    def get_pending_tokens(self) -> int:
        """Get token count for pending messages."""
        text = "\n".join(msg.format() for msg in self._pending_messages)
        return count_tokens(text, self._agent.model_name)


if __name__ == "__main__":
    from llmling_agent import Agent

    async def main():
        async with Agent() as agent:
            await agent.conversation.add_context_from_path("E:/mcp_zed.yml")
            print(agent.conversation.get_history())

    import asyncio

    asyncio.run(main())
