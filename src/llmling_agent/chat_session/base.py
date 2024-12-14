"""Core chat session implementation."""

from __future__ import annotations

from datetime import datetime
import json
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import UUID, uuid4

from pydantic_ai import messages
from pydantic_ai.messages import (
    Message,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolReturn,
)

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
from llmling_agent.commands.output import DefaultOutputWriter
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
        await self._notify_handlers(
            SessionEvent(
                type=SessionEventType.HISTORY_CLEARED,
                timestamp=datetime.now(),
                data={"session_id": str(self.id)},
            )
        )

    async def reset(self) -> None:
        """Reset session state."""
        old_tools = self._tool_states.copy()
        self._history = []
        self._tool_states = self._agent.list_tools()
        await self._notify_handlers(
            SessionEvent(
                type=SessionEventType.SESSION_RESET,
                timestamp=datetime.now(),
                data={
                    "session_id": str(self.id),
                    "previous_tools": old_tools,
                    "new_tools": self._tool_states,
                },
            )
        )

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
        """Send a message and get response(s).

        Args:
            content: Message content to send
            stream: Whether to stream the response
            output: Optional output writer for command responses
            metadata: Optional interface-specific metadata
        """
        if not content.strip():
            msg = "Message cannot be empty"
            raise ValueError(msg)
        if content.startswith("/"):
            # Use provided output or fall back to default
            writer = output or DefaultOutputWriter()
            try:
                await self.handle_command(content[1:], output=writer, metadata=metadata)
                return ChatMessage(content="", role="system")
            except CommandError as e:
                return ChatMessage(content=f"Command error: {e}", role="system")
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
        model_override = self._model if self._model and self._model.strip() else None

        result = await self._agent.run(
            content,
            message_history=self._history,
            model=model_override,  # type: ignore[arg-type]
        )

        # Update history with new messages
        self._history = result.new_messages()
        formatted = self._format_response(result.data)
        model = self._model or self._agent.model_name
        meta = {"tokens": result.cost().total_tokens, "model": model}
        return ChatMessage(content=formatted, role="assistant", metadata=meta)

    def _format_response(self, response: (str | Message)) -> str:  # noqa: PLR0911
        """Format any kind of response in a readable way.

        Args:
            response: Response to format.

        # TODO: Investigate if we should use result.new_messages() instead of
        # result.data for consistency with the streaming interface.

        Returns:
            A human-readable string representation
        """
        match response:
            case str():
                return response
            case ModelTextResponse():
                return response.content
            case ModelStructuredResponse():
                try:
                    calls = [
                        f"Tool: {call.tool_name}\nArgs: {call.args}"
                        for call in response.calls
                    ]
                    return "Tool Calls:\n" + "\n\n".join(calls)
                except Exception as e:  # noqa: BLE001
                    msg = f"Could not format structured response: {e}"
                    logger.warning(msg)
                    return str(response)
            case ToolReturn():
                return f"Tool {response.tool_name} returned: {response.content}"
            case RetryPrompt():
                if isinstance(response.content, str):
                    return f"Retry needed: {response.content}"
                return f"Validation errors:\n{json.dumps(response.content, indent=2)}"
            case _:
                return response.content

    async def _send_streaming(self, content: str) -> AsyncIterator[ChatMessage]:
        """Send message and stream responses."""
        model_override = self._model if self._model and self._model.strip() else None

        async with await self._agent.run_stream(
            content,
            message_history=self._history,
            model=model_override,  # type: ignore[arg-type]
        ) as result:
            # First yield all content chunks
            async for chunk in result.stream():
                content = ""
                match chunk:
                    case messages.ModelTextResponse():
                        content = chunk.content
                    case messages.ToolReturn():
                        content = chunk.model_response_str()
                    case messages.RetryPrompt():
                        content = chunk.model_response()
                    case _:
                        content = str(chunk)
                model = self._model or self._agent.model_name
                meta = {"model": model}
                yield ChatMessage(content=content, role="assistant", metadata=meta)

            # Get cost information, handling both regular and async cases
            cost_result = result.cost()
            cost = await cost_result if hasattr(cost_result, "__await__") else cost_result  # pyright: ignore
            fields = ("total_tokens", "request_tokens", "response_tokens")
            if cost and all(hasattr(cost, attr) for attr in fields):
                usage = {
                    "total": cost.total_tokens,
                    "prompt": cost.request_tokens,
                    "completion": cost.response_tokens,
                }
                model = self._model or self._agent.model_name
                metadata: dict[str, Any] = {"token_usage": usage, "model": model}
                yield ChatMessage(content="", role="assistant", metadata=metadata)

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

        logger.debug("Updated tool states for session %s: %s", self.id, results)
        return results

    def get_tool_states(self) -> dict[str, bool]:
        """Get current tool states."""
        return self._tool_states.copy()

    @property
    def history(self) -> list[messages.Message]:
        """Get conversation history."""
        return list(self._history)
