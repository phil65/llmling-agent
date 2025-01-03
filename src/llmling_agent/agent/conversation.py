"""Conversation management for LLMling agent."""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
import tempfile
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import UUID, uuid4

from llmling import BasePrompt, PromptMessage, StaticPrompt
from llmling.config.models import BaseResource
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from upath import UPath

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent.pydantic_ai_utils import (
    convert_model_message,
    format_response,
    get_message_role,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime

    from llmling.config.models import Resource
    from llmling.prompts import PromptType
    from pydantic_ai.messages import ModelMessage

    from llmling_agent.agent.agent import Agent
    from llmling_agent.common_types import MessageRole, StrPath

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

    def __init__(
        self,
        agent: Agent[Any, Any],
        session_id: str | UUID | None = None,
        initial_prompts: str | Sequence[str] | None = None,
        *,
        resources: Sequence[Resource | str] = (),
    ):
        """Initialize conversation manager.

        Args:
            agent: instance to manage
            session_id: Optional session ID to load and continue conversation
            initial_prompts: Initial system prompts that start each conversation
            resources: Optional paths to load as context
        """
        self._agent = agent
        self._initial_prompts: list[BasePrompt] = []
        self._current_history: list[ModelMessage] = []
        self._last_messages: list[ModelMessage] = []
        self._pending_messages: deque[ModelRequest] = deque()
        if session_id is not None:
            from llmling_agent.storage.models import Message

            # Use provided session ID and load its history
            self.id = str(session_id)
            messages = Message.to_pydantic_ai_messages(self.id)
            self._current_history = messages
        else:
            # Start new conversation with UUID
            self.id = str(uuid4())
            # Add initial prompts
            if not initial_prompts:
                return
            prompts_list = (
                [initial_prompts] if isinstance(initial_prompts, str) else initial_prompts
            )
            for prompt in prompts_list:
                obj = StaticPrompt(
                    name="Initial system prompt",
                    description="Initial system prompt",
                    messages=[PromptMessage(role="system", content=prompt)],
                )
                self._initial_prompts.append(obj)
        # Add context loading tasks to agent
        for source in resources:
            task = asyncio.create_task(self.load_context_source(source))
            self._agent._pending_tasks.add(task)
            task.add_done_callback(self._agent._pending_tasks.discard)

    def __bool__(self) -> bool:
        return bool(self._pending_messages) or bool(self._current_history)

    def __repr__(self) -> str:
        return f"ConversationManager(id={self.id!r})"

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
        from sqlmodel import Session

        from llmling_agent.storage import engine
        from llmling_agent.storage.models import Conversation, Message

        match key:
            case int():
                return convert_model_message(self._current_history[key])
            case slice():
                return [convert_model_message(msg) for msg in self._current_history[key]]
            case str():
                from sqlmodel import or_, select

                # First get all relevant conversation IDs
                stmt = select(Conversation.id).where(
                    or_(
                        Conversation.agent_name == self._agent.name,
                        Conversation.agent_name == key,
                    )
                )

                with Session(engine) as session:
                    conv_ids = session.exec(stmt).all()
                    if not conv_ids:
                        return []  # No conversations found

                    # Now works with sequence of IDs
                    messages = Message.to_pydantic_ai_messages(conv_ids)
                    return [convert_model_message(msg) for msg in messages]

    def __contains__(self, item: Any) -> bool:
        """Check if item is in history."""
        return item in self._current_history

    def __len__(self) -> int:
        """Get length of history."""
        return len(self._current_history)

    def get_message_tokens(self, message: ModelMessage) -> int:
        """Get token count for a single message."""
        import tiktoken

        encoding = tiktoken.encoding_for_model(self._agent.model_name or "gpt-3.5-turbo")
        # Format message to text for token counting
        content = "\n".join(format_response(part) for part in message.parts)
        return len(encoding.encode(content))

    async def format_history(
        self,
        *,
        max_tokens: int | None = None,
        include_system: bool = False,
        format_template: str | None = None,
    ) -> str:
        """Format conversation history as a single context message.

        Args:
            max_tokens: Optional limit to include only last N tokens
            include_system: Whether to include system messages
            format_template: Optional custom format (defaults to agent/message pairs)

        Returns:
            Formatted conversation history as a single string
        """
        template = format_template or "Agent {agent}: {content}\n"
        messages: list[str] = []
        token_count = 0

        history = reversed(self._current_history) if max_tokens else self._current_history

        for msg in history:
            # Check message type instead of role string
            if not include_system and isinstance(msg, SystemPromptPart):
                continue
            content = "\n".join(format_response(part) for part in msg.parts)
            formatted = template.format(agent=get_message_role(msg), content=content)

            if max_tokens:
                # Count tokens in this message
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
        session_id: str | UUID | None = None,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        roles: set[MessageRole] | None = None,
        limit: int | None = None,
    ):
        """Load and set conversation history from database.

        Args:
            session_id: ID of conversation to load
            since: Only include messages after this time
            until: Only include messages before this time
            roles: Only include messages with these roles
            limit: Maximum number of messages to return

        Example:
            # Load last hour of user/assistant messages
            conversation.load_history_from_database(
                "conv-123",
                since=datetime.now() - timedelta(hours=1),
                roles={"user", "assistant"}
            )
        """
        from llmling_agent.storage.models import Message

        match session_id:
            case None:
                session_id = self.id
            case UUID():
                session_id = str(session_id)
        conversation_id = session_id if session_id is not None else self.id
        messages = Message.to_pydantic_ai_messages(
            conversation_id,
            since=since,
            until=until,
            roles=roles,
            limit=limit,
        )
        self.set_history(messages)
        if session_id is not None:
            self.id = session_id

    @asynccontextmanager
    async def temporary(
        self,
        *,
        sys_prompts: PromptInput | Sequence[PromptInput] | None = None,
        mode: OverrideMode = "append",
    ) -> AsyncIterator[None]:
        """Start temporary conversation with different system prompts."""
        # Store original state
        original_prompts = list(self._initial_prompts)
        original_system_prompts = self._agent._pydantic_agent._system_prompts
        original_history = self._current_history

        try:
            if sys_prompts is not None:
                # Convert to list of BasePrompt
                new_prompts: list[BasePrompt] = []
                if isinstance(sys_prompts, str | BasePrompt):
                    new_prompts = [_to_base_prompt(sys_prompts)]
                else:
                    new_prompts = [_to_base_prompt(prompt) for prompt in sys_prompts]

                self._initial_prompts = (
                    original_prompts + new_prompts if mode == "append" else new_prompts
                )

                # Update pydantic-ai's system prompts
                formatted_prompts = await self.get_all_prompts()
                self._agent._pydantic_agent._system_prompts = tuple(formatted_prompts)

            # Force new conversation
            self._current_history = []
            yield
        finally:
            # Restore complete original state
            self._initial_prompts = original_prompts
            self._agent._pydantic_agent._system_prompts = original_system_prompts
            self._current_history = original_history

    def add_prompt(self, prompt: PromptInput):
        """Add a system prompt.

        Args:
            prompt: String content or BasePrompt instance to add
        """
        self._initial_prompts.append(_to_base_prompt(prompt))

    async def get_all_prompts(self) -> list[str]:
        """Get all formatted system prompts in order."""
        result: list[str] = []

        for prompt in self._initial_prompts:
            try:
                messages = await prompt.format()
                result.extend(
                    msg.get_text_content() for msg in messages if msg.role == "system"
                )
            except Exception:
                logger.exception("Error formatting prompt")

        return result

    def get_history(
        self,
        include_pending: bool = True,
        roles: set[type[ModelMessage]] | None = None,
    ) -> list[ModelMessage]:
        """Get current conversation history.

        Args:
            include_pending: Whether to include pending messages in the history.
                             If True, pending messages are moved to main history.
            roles: Message roles to include

        Returns:
            List of messages in chronological order
        """
        if include_pending and self._pending_messages:
            self._current_history.extend(self._pending_messages)
            self._pending_messages.clear()

        if roles:
            return [
                msg
                for msg in self._current_history
                if any(isinstance(msg, r) for r in roles)
            ]
        return self._current_history

    def get_pending_messages(self) -> list[ModelMessage]:
        """Get messages that will be included in next interaction."""
        return list(self._pending_messages)

    def clear_pending(self):
        """Clear pending messages without adding them to history."""
        self._pending_messages.clear()

    def set_history(self, history: list[ModelMessage]):
        """Update conversation history after run."""
        self._current_history = history

    def clear(self):
        """Clear conversation history and prompts."""
        self._initial_prompts.clear()
        self._current_history = []
        self._last_messages = []

    @property
    def last_run_messages(self) -> list[ChatMessage]:
        """Get messages from the last run converted to our format."""
        return [convert_model_message(msg) for msg in self._last_messages]

    async def add_context_message(
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
        message = ModelRequest(parts=[UserPromptPart(content=formatted)])
        self._pending_messages.append(message)
        # Emit as user message - will trigger logging through existing flow

        chat_message = ChatMessage[str](
            content=formatted,
            role="user",
            name="user",
            model=self._agent.model_name,
            metadata=metadata,
        )
        self._agent.message_received.emit(chat_message)

    async def add_context_from_path(
        self,
        path: StrPath,
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
        path_obj = UPath(path)

        if convert_to_md:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()

                # Direct handling for local paths and http(s) URLs
                if path_obj.protocol in ("", "file", "http", "https"):
                    result = md.convert(path_obj.path)
                else:
                    with tempfile.NamedTemporaryFile(suffix=path_obj.suffix) as tmp:
                        tmp.write(path_obj.read_bytes())
                        tmp.flush()
                        result = md.convert(tmp.name)

                content = result.text_content
                source = f"markdown:{path_obj.name}"

            except Exception as e:
                msg = f"Failed to convert {path_obj} to markdown: {e}"
                raise ValueError(msg) from e
        else:
            try:
                content = path_obj.read_text()
                source = f"{path_obj.protocol}:{path_obj.name}"
            except Exception as e:
                msg = f"Failed to read {path_obj}: {e}"
                raise ValueError(msg) from e

        await self.add_context_message(content, source=source, **metadata)

    async def add_context_from_resource(self, resource: Resource | str):
        """Add content from a LLMling resource."""
        if not self._agent.runtime:
            msg = "No runtime available"
            raise RuntimeError(msg)

        if isinstance(resource, str):
            content = await self._agent.runtime.load_resource(resource)
            await self.add_context_message(
                str(content.content),
                source=f"Resource {resource}",
                mime_type=content.metadata.mime_type,
                **content.metadata.extra,
            )
        else:
            loader = self._agent.runtime._loader_registry.get_loader(resource)
            async for content in loader.load(resource):
                await self.add_context_message(
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

            await self.add_context_message(
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
        import tiktoken

        encoding = tiktoken.encoding_for_model(self._agent.model_name or "gpt-3.5-turbo")
        return sum(
            len(encoding.encode(format_response(part)))
            for msg in self._current_history
            for part in msg.parts
        )

    def get_pending_tokens(self) -> int:
        """Get token count for pending messages."""
        import tiktoken

        encoding = tiktoken.encoding_for_model(self._agent.model_name or "gpt-3.5-turbo")
        return sum(
            len(encoding.encode(format_response(part)))
            for msg in self._pending_messages
            for part in msg.parts
        )


if __name__ == "__main__":
    from llmling_agent import Agent

    async def main():
        async with Agent[Any, Any].open() as agent:
            convo = ConversationManager(agent, session_id="test")
            await convo.add_context_from_path("E:/mcp_zed.yml")
            print(convo._current_history)

    import asyncio

    asyncio.run(main())
