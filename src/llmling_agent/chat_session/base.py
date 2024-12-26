"""Core chat session implementation."""

from __future__ import annotations

from datetime import datetime
import pathlib
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import UUID, uuid4

from platformdirs import user_data_dir
from psygnal import Signal
from slashed import (
    BaseCommand,
    CommandError,
    CommandStore,
    DefaultOutputWriter,
    ExitCommandError,
)

from llmling_agent import LLMlingAgent
from llmling_agent.chat_session.events import (
    HistoryClearedEvent,
    SessionResetEvent,
)
from llmling_agent.chat_session.exceptions import ChatSessionConfigError
from llmling_agent.chat_session.models import ChatSessionMetadata, SessionState
from llmling_agent.commands import get_commands
from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage, MessageMetadata
from llmling_agent.pydantic_ai_utils import extract_usage
from llmling_agent.storage.models import CommandHistory
from llmling_agent.tools.base import ToolInfo


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import messages

    from llmling_agent.chat_session.output import OutputWriter
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.snippets import Snippet
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)
HISTORY_DIR = pathlib.Path(user_data_dir("llmling", "llmling")) / "cli_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class AgentChatSession:
    """Manages an interactive chat session with an agent.

    This class:
    1. Manages agent configuration (tools, model)
    2. Handles conversation flow
    3. Tracks session state and metadata
    """

    history_cleared = Signal(HistoryClearedEvent)
    session_reset = Signal(SessionResetEvent)
    tool_added = Signal(ToolInfo)
    tool_removed = Signal(str)  # tool_name
    tool_changed = Signal(str, ToolInfo)  # name, new_info
    agent_connected = Signal(LLMlingAgent)

    def __init__(
        self,
        agent: LLMlingAgent[Any, str],
        *,
        pool: AgentPool | None = None,
        wait_chain: bool = True,
        session_id: UUID | str | None = None,
        model_override: str | None = None,
    ):
        """Initialize chat session.

        Args:
            agent: The LLMling agent to use
            pool: Optional agent pool for multi-agent interactions
            wait_chain: Whether to wait for chain completion
            session_id: Optional session ID (generated if not provided)
            model_override: Optional model override for this session
        """
        # Basic setup that doesn't need async
        match session_id:
            case str():
                self.id = UUID(session_id)
            case UUID():
                self.id = session_id
            case None:
                self.id = uuid4()
        self._agent = agent
        self._pool = pool
        self._wait_chain = wait_chain
        # forward ToolManager signals to ours
        self._tool_states = self._agent.tools.list_tools()
        self._agent.tools.events.added.connect(self.tool_added.emit)
        self._agent.tools.events.removed.connect(self.tool_removed.emit)
        self._agent.tools.events.changed.connect(self.tool_changed.emit)
        self._model = model_override or agent.model_name
        self._history: list[messages.ModelMessage] = []
        self._commands: list[str] = []
        self._history_file = HISTORY_DIR / f"{agent.name}.history"
        self._initialized = False  # Track initialization state

        # Initialize basic structures
        self.commands = CommandStore()
        self.start_time = datetime.now()
        self._state = SessionState(current_model=self._model)

    @property
    def pool(self) -> AgentPool | None:
        """Get the agent pool if available."""
        return self._pool

    @property
    def wait_chain(self) -> bool:
        """Whether to wait for chain completion."""
        return self._wait_chain

    @wait_chain.setter
    def wait_chain(self, value: bool):
        """Set chain waiting behavior."""
        if not isinstance(value, bool):
            msg = f"wait_chain must be bool, got {type(value)}"
            raise TypeError(msg)
        self._wait_chain = value

    async def connect_to(self, target: str, wait: bool | None = None) -> None:
        """Connect to another agent.

        Args:
            target: Name of target agent
            wait: Override session's wait_chain setting

        Raises:
            ValueError: If target agent not found or pool not available
        """
        logger.debug("Connecting to %s (wait=%s)", target, wait)
        if not self._pool:
            msg = "No agent pool available"
            raise ValueError(msg)

        try:
            target_agent = self._pool.get_agent(target)
        except KeyError as e:
            msg = f"Target agent not found: {target}"
            raise ValueError(msg) from e

        self._agent.pass_results_to(target_agent)
        self.agent_connected.emit(target_agent)

        if wait is not None:
            self._wait_chain = wait

    async def disconnect_from(self, target: str) -> None:
        """Disconnect from a target agent."""
        if not self._pool:
            msg = "No agent pool available"
            raise ValueError(msg)

        target_agent = self._pool.get_agent(target)
        self._agent.stop_passing_results_to(target_agent)

    async def disconnect_all(self) -> None:
        """Disconnect from all agents."""
        if self._agent._connected_agents:
            connected = list(self._agent._connected_agents)
            for target in connected:
                self._agent.stop_passing_results_to(target)

    def get_connections(self) -> list[tuple[str, bool]]:
        """Get current connections.

        Returns:
            List of (agent_name, waits_for_completion) tuples
        """
        return [(agent.name, self._wait_chain) for agent in self._agent._connected_agents]

    def _ensure_initialized(self):
        """Check if session is initialized."""
        if not self._initialized:
            msg = "Session not initialized. Call initialize() first."
            raise RuntimeError(msg)

    async def initialize(self):
        """Initialize async resources and load data."""
        if self._initialized:
            return

        # Load command history
        self._load_commands()
        # Initialize command system
        self.commands.register_builtin_commands()
        for cmd in get_commands():
            self.commands.register_command(cmd)

        self._initialized = True
        msg = "Initialized chat session %r for agent %r"
        logger.debug(msg, self.id, self._agent.name)

    async def cleanup(self):
        """Clean up session resources."""
        if self._pool:
            await self.disconnect_all()

    def add_snippet(self, content: str, source: str) -> Snippet:
        """Add content to be included in next prompt."""
        return self._agent.snippets.add_snippet(content, source=source)

    def _load_commands(self):
        """Load command history from file."""
        try:
            if self._history_file.exists():
                self._commands = self._history_file.read_text().splitlines()
        except Exception:
            logger.exception("Failed to load command history")
            self._commands = []

    def add_command(self, command: str):
        """Add command to history."""
        if not command.strip():
            return
        CommandHistory.log(self._agent.name, str(self.id), command)

    def get_commands(
        self, limit: int | None = None, current_session_only: bool = False
    ) -> list[str]:
        """Get command history ordered by newest first."""
        return CommandHistory.get_commands(
            agent_name=self._agent.name,
            session_id=str(self.id),
            limit=limit,
            current_session_only=current_session_only,
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

    async def clear(self):
        """Clear chat history."""
        self._history = []
        event = HistoryClearedEvent(session_id=str(self.id))
        self.history_cleared.emit(event)

    async def reset(self):
        """Reset session state."""
        old_tools = self._tool_states.copy()
        self._history = []
        self._tool_states = self._agent.tools.list_tools()

        event = SessionResetEvent(
            session_id=str(self.id),
            previous_tools=old_tools,
            new_tools=self._tool_states,
        )
        self.session_reset.emit(event)

    def register_command(self, command: BaseCommand):
        """Register additional command."""
        self.commands.register_command(command)

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
        self._ensure_initialized()
        meta = metadata or {}
        ctx = self.commands.create_context(self, output_writer=output, metadata=meta)
        await self.commands.execute_command(command_str, ctx)

    async def send_slash_command(
        self,
        content: str,
        *,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        writer = output or DefaultOutputWriter()
        try:
            await self.handle_command(content[1:], output=writer, metadata=metadata)
            return ChatMessage(content="", role="system")
        except ExitCommandError:
            # Re-raise without wrapping in CommandError
            raise
        except CommandError as e:
            return ChatMessage(content=f"Command error: {e}", role="system")

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
        self._ensure_initialized()
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

        result = await self._agent.run(
            content,
            message_history=self._history,
            model=model_override,  # type: ignore
        )

        # Update history with new messages
        self._history = result.new_messages()

        model_name = model_override or self._agent.model_name
        response = str(result.data)
        cost_info = (
            await extract_usage(result.usage(), model_name, content, response)
            if model_name
            else None
        )

        metadata = {}
        if cost_info:
            metadata.update({
                "token_usage": cost_info.token_usage,
                "cost_usd": cost_info.cost_usd,
            })
        if model_name:
            metadata["model"] = model_name

        # Update session state before returning
        self._state.message_count += 2  # User and assistant messages
        usage = cost_info.token_usage if cost_info else None
        cost = cost_info.cost_usd if cost_info else None
        metadata_obj = MessageMetadata(model=model_name, token_usage=usage, cost=cost)

        chat_msg: ChatMessage[str] = ChatMessage(
            content=response,
            role="assistant",
            metadata=metadata_obj,
            token_usage=metadata_obj.token_usage,
        )
        self._state.update_tokens(chat_msg)
        # Add chain waiting if enabled
        if self._wait_chain and self._pool:
            await self._agent.wait_for_chain()

        return chat_msg

    async def _stream_message(
        self,
        content: str,
    ) -> AsyncIterator[ChatMessage]:
        """Send message and stream responses."""
        async with self._agent.run_stream(
            content,
            message_history=self._history,
            model=self._model or "",  # type: ignore
        ) as stream_result:
            async for response in stream_result.stream():
                yield ChatMessage[str](content=str(response), role="assistant")

            # Final message with token usage after stream completes
            metadata: dict[str, Any] = {}
            if model_name := self._model or self._agent.model_name:
                metadata["model"] = model_name
                usage = stream_result.usage()
                if cost_info := await extract_usage(usage, model_name, content, response):
                    metadata.update({
                        "token_usage": cost_info.token_usage,
                        "cost_usd": cost_info.cost_usd,
                    })
            # Update session state after stream completes
            self._state.message_count += 2  # User and assistant messages
            meta_obj = MessageMetadata(**metadata)
            final_msg: ChatMessage[str] = ChatMessage(
                content="",  # Empty content for final status message
                role="assistant",
                metadata=meta_obj,
                token_usage=meta_obj.token_usage,
            )
            self._state.update_tokens(final_msg)

            # Add chain waiting if enabled
            if self._wait_chain and self._pool:
                await self._agent.wait_for_chain()

            yield final_msg

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

    def has_chain(self) -> bool:
        """Check if agent has any connections."""
        return bool(self._agent._connected_agents)

    def is_processing_chain(self) -> bool:
        """Check if chain is currently processing."""
        return any(a._pending_tasks for a in self._agent._connected_agents)

    def get_tool_states(self) -> dict[str, bool]:
        """Get current tool states."""
        return self._agent.tools.list_tools()

    @property
    def tools(self) -> ToolManager:
        """Get current tool states."""
        return self._agent.tools

    @property
    def history(self) -> list[messages.ModelMessage]:
        """Get conversation history."""
        return list(self._history)

    def get_status(self) -> SessionState:
        """Get current session status."""
        return self._state
