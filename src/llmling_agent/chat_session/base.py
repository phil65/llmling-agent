"""Core chat session implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import UUID, uuid4

from llmling_agent.chat_session.events import (
    SessionEvent,
    SessionEventHandler,
    SessionEventType,
)
from llmling_agent.chat_session.exceptions import ChatSessionConfigError
from llmling_agent.chat_session.models import ChatMessage, ChatSessionMetadata
from llmling_agent.commands import CommandStore
from llmling_agent.commands.base import (
    BaseCommand,
    CommandContext,
    CommandError,
    OutputWriter,
)
from llmling_agent.commands.exceptions import ExitCommandError
from llmling_agent.commands.output import DefaultOutputWriter
from llmling_agent.log import get_logger
from llmling_agent.pydantic_ai_utils import extract_token_usage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import messages
    from pydantic_ai.result import RunResult

    from llmling_agent import LLMlingAgent


logger = get_logger(__name__)


class AgentChatSession:
    """Manages an interactive chat session with an agent.

    This class:
    1. Manages agent configuration (tools, model)
    2. Handles conversation flow
    3. Tracks session state and metadata
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
        self._history: list[messages.ModelMessage] = []
        self._tool_states = agent.tools.list_tools()
        self._model = model_override or agent.model_name
        msg = "Created chat session %s for agent %s"
        logger.debug(msg, self.id, agent.name)
        # Initialize command system
        self._command_store = CommandStore()
        self._command_store.register_builtin_commands()
        self._event_handlers: list[SessionEventHandler] = []

    @property
    def metadata(self) -> ChatSessionMetadata:
        """Get current session metadata."""
        return ChatSessionMetadata(
            session_id=self.id,
            agent_name=self._agent.name,
            model=self._model,
            tool_states=self._tool_states,
        )

    def add_event_handler(self, handler: SessionEventHandler) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: SessionEventHandler) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def _notify_handlers(self, event: SessionEvent) -> None:
        """Notify all handlers of an event."""
        for handler in self._event_handlers:
            await handler.handle_session_event(event)

    async def clear(self) -> None:
        """Clear chat history."""
        self._history = []
        data = {"session_id": str(self.id)}
        event = SessionEvent(type=SessionEventType.HISTORY_CLEARED, data=data)
        await self._notify_handlers(event)

    async def reset(self) -> None:
        """Reset session state."""
        old_tools = self._tool_states.copy()
        self._history = []
        self._tool_states = self._agent.tools.list_tools()
        data = {
            "session_id": str(self.id),
            "previous_tools": old_tools,
            "new_tools": self._tool_states,
        }
        event = SessionEvent(type=SessionEventType.SESSION_RESET, data=data)
        await self._notify_handlers(event)

    def register_command(self, command: BaseCommand) -> None:
        """Register additional command."""
        self._command_store.register_command(command)

    async def handle_command(
        self,
        command_str: str,
        output: OutputWriter,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Handle a slash command.

        Args:
            command_str: Command string without leading slash
            output: Output writer implementation
            metadata: Optional interface-specific metadata
        """
        ctx = CommandContext(
            output=output,
            session=self,
            metadata=metadata or {},
        )
        await self._command_store.execute_command(command_str, ctx)

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[False] = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage: ...

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[True],
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatMessage]: ...

    async def send_message(
        self,
        content: str,
        *,
        stream: bool = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage | AsyncIterator[ChatMessage]:
        """Send a message and get response(s)."""
        if not content.strip():
            msg = "Message cannot be empty"
            raise ValueError(msg)

        if content.startswith("/"):
            writer = output or DefaultOutputWriter()
            try:
                await self.handle_command(content[1:], output=writer, metadata=metadata)
                return ChatMessage(content="", role="system")
            except ExitCommandError:
                # Re-raise without wrapping in CommandError
                raise
            except CommandError as e:
                return ChatMessage(content=f"Command error: {e}", role="system")

        try:
            # Update tool states in pydantic agent before call
            self._agent._pydantic_agent._function_tools.clear()
            enabled_tools = self._agent.tools.get_tools(state="enabled")
            for tool in enabled_tools:
                assert tool._original_callable
                self._agent._pydantic_agent.tool_plain(tool._original_callable)

            if stream:
                return self._stream_message(content)
            return await self._send_normal(content)

        except Exception as e:
            logger.exception("Error processing message")
            msg = f"Error processing message: {e}"
            raise ChatSessionConfigError(msg) from e

    async def _send_normal(self, content: str) -> ChatMessage:
        """Send message and get single response."""
        model_override = self._model if self._model and self._model.strip() else None

        result: RunResult = await self._agent.run(
            content,
            message_history=self._history,
            model=model_override,  # type: ignore
        )

        # Update history with new messages
        self._history = result.new_messages()
        usage = extract_token_usage(result.cost())
        meta = {"token_usage": usage, "model": self._agent.model_name}
        return ChatMessage(content=str(result.data), role="assistant", metadata=meta)

    async def _stream_message(self, content: str) -> AsyncIterator[ChatMessage]:
        """Send message and stream responses."""
        model_override = self._model if self._model and self._model.strip() else None

        async with await self._agent.run_stream(
            content,
            message_history=self._history,
            model=model_override or "",  # type: ignore
        ) as stream_result:
            async for response in stream_result.stream():
                meta = {"model": self._agent.model_name}
                yield ChatMessage(content=str(response), role="assistant", metadata=meta)

            # Final message with token usage after stream completes
            cost = stream_result.cost()
            usage = extract_token_usage(cost)
            metadata = {"token_usage": usage, "model": self._agent.model_name}
            yield ChatMessage(content="", role="assistant", metadata=metadata)

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
                    self._agent.tools.enable_tool(tool)
                    results[tool] = "enabled"
                else:
                    self._agent.tools.disable_tool(tool)
                    results[tool] = "disabled"
                self._tool_states[tool] = enabled
            except ValueError as e:
                results[tool] = f"error: {e}"

        logger.debug("Updated tool states for session %s: %s", self.id, results)
        return results

    def get_tool_states(self) -> dict[str, bool]:
        """Get current tool states."""
        return self._agent.tools.list_tools()

    @property
    def history(self) -> list[messages.ModelMessage]:
        """Get conversation history."""
        return list(self._history)
