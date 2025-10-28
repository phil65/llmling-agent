"""Core chat session implementation."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal, overload

from platformdirs import user_data_dir
from psygnal import Signal

from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.tools.base import Tool
from llmling_agent.utils.now import get_now
from llmling_agent_cli.chat_session.exceptions import ChatSessionConfigError
from llmling_agent_cli.chat_session.models import SessionState
from llmling_agent_commands import get_commands


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from slashed import OutputWriter

    from llmling_agent import AgentPool, AnyAgent
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)
HISTORY_DIR = pathlib.Path(user_data_dir("llmling", "llmling")) / "cli_history"


class AgentPoolView:
    """Helper class for agent (pool)."""

    history_cleared = Signal(ConversationManager.HistoryCleared)
    tool_added = Signal(str, Tool)
    tool_removed = Signal(str)  # tool_name
    tool_changed = Signal(str, Tool)  # name, new_info

    def __init__(
        self,
        agent: AnyAgent[Any, Any],
        *,
        pool: AgentPool | None = None,
    ):
        """Initialize chat session.

        Args:
            agent: The LLMling agent to use
            pool: Optional agent pool for multi-agent interactions
        """
        from slashed import CommandStore

        self._agent = agent
        self.pool = pool
        self._agent.tools.events.added.connect(self.tool_added)
        self._agent.tools.events.removed.connect(self.tool_removed)
        self._agent.tools.events.changed.connect(self.tool_changed)
        self._agent.conversation.history_cleared.connect(self.history_cleared)
        file_path = HISTORY_DIR / f"{agent.name}.history"
        self.commands = CommandStore(history_file=file_path, enable_system_commands=True)
        self.start_time = get_now()
        self._state = SessionState(current_model=self._agent.model_name)
        # Store wait state for this connection
        self.commands._initialize_sync()
        for cmd in get_commands():
            self.commands.register_command(cmd)
        logger.debug("Initialized chat session for agent %r", self._agent.name)

    async def connect_to(self, target: str, wait: bool | None = None):
        """Connect to another agent.

        Args:
            target: Name of target agent
            wait: Override session's wait_chain setting

        Raises:
            ValueError: If target agent not found or pool not available
        """
        assert self.pool
        target_agent = self.pool.get_agent(target)
        self._agent.connect_to(target_agent)
        self._agent.connections.set_wait_state(target, wait if wait is not None else True)

    async def cleanup(self):
        """Clean up session resources."""
        if self.pool:
            await self._agent.disconnect_all()

    def add_command(self, command: str):
        """Add command to history."""
        if not command.strip():
            return

        id_ = str(self._agent.conversation.id)
        self._agent.context.storage.log_command.sync(
            agent_name=self._agent.name, session_id=id_, command=command
        )

    def get_commands(
        self, limit: int | None = None, current_session_only: bool = False
    ) -> list[str]:
        """Get command history ordered by newest first."""
        return self._agent.context.storage.get_commands.sync(
            agent_name=self._agent.name,
            session_id=str(self._agent.conversation.id),
            limit=limit,
            current_session_only=current_session_only,
        )

    async def handle_command(
        self,
        command_str: str,
        output: OutputWriter,
        metadata: dict[str, Any] | None = None,
    ):
        """Handle a slash command.

        Args:
            command_str: Command string without leading slash
            output: Output writer implementation
            metadata: Optional interface-specific metadata
        """
        meta = metadata or {}
        ctx = self.commands.create_context(
            self._agent.context,
            output_writer=output,
            metadata=meta,
        )
        await self.commands.execute_command(command_str, ctx)

    async def send_slash_command(
        self,
        content: str,
        *,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]:
        from slashed import CommandError, DefaultOutputWriter, ExitCommandError

        writer = output or DefaultOutputWriter()
        try:
            await self.handle_command(content[1:], output=writer, metadata=metadata)
            return ChatMessage(content="", role="system", name="System")
        except ExitCommandError:
            # Re-raise without wrapping in CommandError
            raise
        except CommandError as e:
            return ChatMessage(
                content=f"Command error: {e}", role="system", name="System"
            )

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[False] = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]: ...

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[True],
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatMessage[str]]: ...

    async def send_message(
        self,
        content: str,
        *,
        stream: bool = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str] | AsyncIterator[ChatMessage[str]]:
        """Send a message and get response(s)."""
        if not content.strip():
            msg = "Message cannot be empty"
            raise ValueError(msg)

        if content.startswith("/"):
            return await self.send_slash_command(
                content,
                output=output,
                metadata=metadata,
            )
        try:
            if stream:
                return self._stream_message(content)
            return await self._send_normal(content)

        except Exception as e:
            logger.exception("Error processing message")
            msg = f"Error processing message: {e}"
            raise ChatSessionConfigError(msg) from e

    async def _send_normal(self, content: str) -> ChatMessage[str]:
        """Send message and get single response."""
        result = await self._agent.run(content)
        text_message = result.to_text_message()

        # Update session state metrics
        self._state.message_count += 2  # User and assistant messages
        if text_message.cost_info:
            self._state.update_tokens(text_message)
            self._state.total_cost = float(text_message.cost_info.total_cost)
        if text_message.response_time:
            self._state.last_response_time = text_message.response_time
        return text_message

    async def _stream_message(self, content: str) -> AsyncIterator[ChatMessage[str]]:
        """Send message and stream responses."""
        from pydantic_ai import PartDeltaEvent, TextPartDelta

        from llmling_agent.agent.agent import StreamCompleteEvent

        async for event in self._agent.run_stream(content):
            match event:
                case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                    yield ChatMessage[str](
                        content=delta,
                        role="assistant",
                        name=self._agent.name,
                    )
                case StreamCompleteEvent(message=message):
                    # Use the embedded final message for metrics
                    final_msg = message

            # Update session state
            self._state.message_count += 2  # User and assistant messages
            self._state.update_tokens(final_msg)

            yield final_msg

    @property
    def tools(self) -> ToolManager:
        """Get current tool states."""
        return self._agent.tools
