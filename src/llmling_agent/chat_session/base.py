"""Core chat session implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from llmling.prompts import PromptMessage
from pydantic_ai import messages

from llmling_agent.chat_session.commands import CommandRegistry
from llmling_agent.chat_session.exceptions import ChatSessionConfigError
from llmling_agent.chat_session.models import ChatMessage, ChatSessionMetadata
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent import LLMlingAgent


logger = get_logger(__name__)


class AgentChatSession:
    """Manages an interactive chat session with an agent.

    This class:
    1. Manages agent configuration (tools, model)
    2. Handles conversation flow
    3. Tracks session state and metadata
    4. Processes slash commands
    """

    def __init__(
        self,
        agent: LLMlingAgent[str],
        *,
        session_id: UUID | None = None,
        model_override: str | None = None,
    ) -> None:
        """Initialize chat session.

        Args:
            agent: The LLMling agent to use
            session_id: Optional session ID (generated if not provided)
            model_override: Optional model override for this session
        """
        self.id = session_id or uuid4()
        self._agent = agent
        self._history: list[messages.Message] = []
        self._tool_states = agent.list_tools()
        self._model = model_override

        # Initialize command registry
        self._command_registry = CommandRegistry(self)

        logger.debug(
            "Created chat session %s for agent %s",
            self.id,
            agent.name,
        )

    @property
    def metadata(self) -> ChatSessionMetadata:
        """Get current session metadata."""
        return ChatSessionMetadata(
            session_id=self.id,
            agent_name=self._agent.name,
            model=self._model,
            tool_states=self._tool_states,
        )

    @property
    def runtime(self):
        """Get the runtime configuration."""
        return self._agent.runtime

    @property
    def capabilities(self):
        """Get agent capabilities."""
        return self._agent._context.capabilities

    async def process_message(
        self,
        content: str,
        *,
        stream: bool = False,
    ) -> ChatMessage | AsyncIterator[ChatMessage]:
        """Process a message, handling commands or normal messages.

        Args:
            content: Message content
            stream: Whether to stream the response

        Returns:
            Response message or stream of message chunks
        """
        # Handle slash commands
        if content.startswith("/"):
            result = await self._command_registry.execute(content[1:])
            return ChatMessage(
                content=result["content"],
                role="system",
                metadata=result["metadata"],
            )

        # Normal message handling
        return await self.send_message(content, stream=stream)

    async def send_message(
        self,
        content: str | PromptMessage,
        *,
        stream: bool = False,
    ) -> ChatMessage | AsyncIterator[ChatMessage]:
        """Send a message and get response(s).

        Args:
            content: Message content or PromptMessage
            stream: Whether to stream the response

        Returns:
            Either a single message or an async iterator of message chunks

        Raises:
            ChatSessionConfigError: If message processing fails
        """
        # Convert content to string if it's a PromptMessage
        if isinstance(content, PromptMessage):
            content = content.get_text_content()

        self._history.append(messages.UserPrompt(content=content))

        try:
            if stream:
                return self._send_streaming(content)
            return await self._send_normal(content)
        except Exception as e:
            logger.exception("Error processing message")
            msg = f"Error processing message: {e}"
            raise ChatSessionConfigError(msg) from e

    async def _send_normal(self, content: str) -> ChatMessage:
        """Send message and get single response."""
        result = await self._agent.run(
            content,
            message_history=self._history,
            model=self._model,
        )

        # Update history with new messages
        self._history = result.new_messages()

        return ChatMessage(
            content=str(result.data),
            role="assistant",
            metadata={
                "tokens": result.cost().total_tokens,
                "model": self._model or str(self._agent._pydantic_agent.model),
            },
        )

    async def _send_streaming(self, content: str) -> AsyncIterator[ChatMessage]:
        """Send message and stream responses."""
        async with await self._agent.run_stream(
            content,
            message_history=self._history,
            model=self._model,
        ) as result:
            async for chunk in result.stream():
                yield ChatMessage(
                    content=str(chunk),
                    role="assistant",
                )
            # Update history after stream completes
            self._history = result.new_messages()

    def configure_tools(
        self,
        updates: dict[str, bool],
    ) -> dict[str, str]:
        """Update tool configuration.

        Args:
            updates: Mapping of tool names to desired states

        Returns:
            Mapping of tool names to status messages
        """
        results = {}
        for tool, enabled in updates.items():
            try:
                if enabled:
                    self._agent.enable_tool(tool)
                    results[tool] = "enabled"
                else:
                    self._agent.disable_tool(tool)
                    results[tool] = "disabled"
                self._tool_states[tool] = enabled
            except ValueError as e:
                results[tool] = f"error: {e}"

        logger.debug(
            "Updated tool states for session %s: %s",
            self.id,
            results,
        )
        return results

    def get_tool_states(self) -> dict[str, bool]:
        """Get current tool states."""
        return self._tool_states.copy()

    @property
    def history(self) -> list[messages.Message]:
        """Get conversation history."""
        return list(self._history)
