"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from anyenv import MultiEventHandler
from anyenv.signals import BoundSignal
from exxec import LocalExecutionEnvironment

from agentpool.agents.events import resolve_event_handlers
from agentpool.log import get_logger
from agentpool.messaging import MessageHistory, MessageNode
from agentpool.tools.manager import ToolManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment
    from slashed import BaseCommand, CommandStore
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp.schema import AvailableCommandsUpdate, ConfigOptionUpdate
    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory, ModeInfo
    from agentpool.common_types import BuiltinEventHandlerType, IndividualEventHandler
    from agentpool.delegation import AgentPool
    from agentpool.talk.stats import MessageStats
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig

    # Union type for state updates emitted via state_updated signal
    type StateUpdate = ModeInfo | ModelInfo | AvailableCommandsUpdate | ConfigOptionUpdate


logger = get_logger(__name__)


ToolConfirmationMode = Literal["always", "never", "per_tool"]


class BaseAgent[TDeps = None, TResult = str](MessageNode[TDeps, TResult]):
    """Base class for Agent, ACPAgent, AGUIAgent, and ClaudeCodeAgent.

    Provides shared infrastructure:
    - tools: ToolManager for tool registration and execution
    - conversation: MessageHistory for conversation state
    - event_handler: MultiEventHandler for event distribution
    - _event_queue: Queue for streaming events
    - tool_confirmation_mode: Tool confirmation behavior
    - _input_provider: Provider for user input/confirmations
    - env: ExecutionEnvironment for running code/commands
    - context property: Returns NodeContext for the agent
    """

    def __init__(
        self,
        *,
        name: str = "agent",
        description: str | None = None,
        display_name: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        # New shared parameters
        env: ExecutionEnvironment | None = None,
        input_provider: InputProvider | None = None,
        output_type: type[TResult] = str,  # type: ignore[assignment]
        tool_confirmation_mode: ToolConfirmationMode = "per_tool",
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        commands: Sequence[BaseCommand] | None = None,
    ) -> None:
        """Initialize base agent with shared infrastructure.

        Args:
            name: Agent name
            description: Agent description
            display_name: Human-readable display name
            mcp_servers: MCP server configurations
            agent_pool: Agent pool for coordination
            enable_logging: Whether to enable database logging
            event_configs: Event trigger configurations
            env: Execution environment for running code/commands
            input_provider: Provider for user input and confirmations
            output_type: Output type for this agent
            tool_confirmation_mode: How tool execution confirmation is handled
            event_handlers: Event handlers for this agent
            commands: Slash commands to register with this agent
        """
        super().__init__(
            name=name,
            description=description,
            display_name=display_name,
            mcp_servers=mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs,
        )

        # Shared infrastructure - previously duplicated in all 4 agents
        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = asyncio.Queue()
        self.conversation = MessageHistory()
        self.env = env or LocalExecutionEnvironment()
        self._input_provider = input_provider
        self._output_type: type[TResult] = output_type
        self.tool_confirmation_mode: ToolConfirmationMode = tool_confirmation_mode
        self.tools = ToolManager()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(
            resolved_handlers
        )

        # Cancellation infrastructure
        self._cancelled = False
        self._current_stream_task: asyncio.Task[Any] | None = None

        # State change signal - emitted when mode/model/commands change
        # Uses union type for different state update kinds
        self.state_updated: BoundSignal[StateUpdate] = BoundSignal()

        # Command store for slash commands
        from slashed import CommandStore

        self._command_store: CommandStore = CommandStore()

        # Register provided commands
        if commands:
            for command in commands:
                self._command_store.register_command(command)

    @property
    def command_store(self) -> CommandStore:
        """Get the command store for slash commands."""
        return self._command_store

    @abstractmethod
    def get_context(self, data: Any = None) -> AgentContext[Any]:
        """Create a new context for this agent.

        Args:
            data: Optional custom data to attach to the context

        Returns:
            A new AgentContext instance
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str | None:
        """Get the model name used by this agent."""
        ...

    @abstractmethod
    async def set_model(self, model: str) -> None:
        """Set the model for this agent.

        Args:
            model: New model identifier to use
        """
        ...

    @abstractmethod
    def run_stream(
        self,
        *prompt: Any,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Run agent with streaming output.

        Args:
            *prompt: Input prompts
            **kwargs: Additional arguments

        Yields:
            Stream events during execution
        """
        ...

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode.

        Args:
            mode: Confirmation mode - "always", "never", or "per_tool"
        """
        self.tool_confirmation_mode = mode

    def is_cancelled(self) -> bool:
        """Check if the agent has been cancelled.

        Returns:
            True if cancellation was requested
        """
        return self._cancelled

    async def interrupt(self) -> None:
        """Interrupt the currently running stream.

        This method is called when cancellation is requested. The default
        implementation sets the cancelled flag and cancels the current stream task.

        Subclasses may override to add protocol-specific cancellation:
        - ACPAgent: Send CancelNotification to remote server
        - ClaudeCodeAgent: Call client.interrupt()

        The cancelled flag should be checked in run_stream loops to exit early.
        """
        self._cancelled = True
        if self._current_stream_task and not self._current_stream_task.done():
            self._current_stream_task.cancel()
            logger.info("Interrupted agent stream", agent=self.name)

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        from agentpool.talk.stats import MessageStats

        return MessageStats(messages=list(self.conversation.chat_messages))

    @abstractmethod
    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models for this agent.

        Returns a list of models that can be used with this agent, or None
        if model discovery is not supported for this agent type.

        Uses tokonomics.ModelInfo which includes pricing, capabilities,
        and limits. Can be converted to protocol-specific formats (OpenCode, ACP).

        Returns:
            List of tokonomics ModelInfo, or None if not supported
        """
        ...

    @abstractmethod
    def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for this agent.

        Returns a list of mode categories that can be switched. Each category
        represents a group of mutually exclusive modes (e.g., permissions,
        behavior presets).

        Different agent types expose different modes:
        - Native Agent: Tool confirmation modes (default, acceptEdits)
        - ClaudeCodeAgent: Claude Code SDK modes (plan, code, etc.)
        - ACPAgent: Passthrough from remote server
        - AGUIAgent: Empty list (no modes)

        Returns:
            List of ModeCategory, empty list if no modes supported
        """
        ...

    @abstractmethod
    async def set_mode(self, mode: ModeInfo | str, category_id: str | None = None) -> None:
        """Set a mode within a category.

        Each agent type handles mode switching according to its own semantics:
        - Native Agent: Maps to tool confirmation mode
        - ClaudeCodeAgent: Maps to SDK permission mode
        - ACPAgent: Forwards to remote server
        - AGUIAgent: No-op (no modes supported)

        Args:
            mode: The mode to activate - either a ModeInfo object or mode ID string.
                  If ModeInfo, category_id is extracted from it (unless overridden).
            category_id: Optional category ID. If None and mode is a string,
                         uses the first category. If None and mode is ModeInfo,
                         uses the mode's category_id.

        Raises:
            ValueError: If mode_id or category_id is invalid
        """
        ...
