"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from anyenv import MultiEventHandler, method_spawner
from anyenv.signals import BoundSignal

from agentpool.agents.events import resolve_event_handlers
from agentpool.log import get_logger
from agentpool.messaging import MessageHistory, MessageNode
from agentpool.tools.manager import ToolManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import UserContent
    from slashed import BaseCommand, CommandStore
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp.schema import AvailableCommandsUpdate, ConfigOptionUpdate
    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory, ModeInfo
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        IndividualEventHandler,
        MCPServerStatus,
        PromptCompatible,
    )
    from agentpool.delegation import AgentPool
    from agentpool.messaging import ChatMessage
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
        from exxec import LocalExecutionEnvironment
        from slashed import CommandStore

        from agentpool_commands import get_commands

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
        # Use storage from agent_pool if available, otherwise memory-only
        storage = agent_pool.storage if agent_pool else None
        self.conversation = MessageHistory(storage=storage)
        self.env = env or LocalExecutionEnvironment()
        self._input_provider = input_provider
        self._output_type: type[TResult] = output_type
        self.tool_confirmation_mode: ToolConfirmationMode = tool_confirmation_mode
        self.tools = ToolManager()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(
            resolved_handlers
        )
        self._cancelled = False
        self._current_stream_task: asyncio.Task[Any] | None = None
        # Deferred initialization support - subclasses set True in __aenter__,
        # override ensure_initialized() to do actual connection
        self._connect_pending: bool = False
        # State change signal - emitted when mode/model/commands change
        # Uses union type for different state update kinds
        self.state_updated: BoundSignal[StateUpdate] = BoundSignal()
        self._command_store: CommandStore = CommandStore()
        # Initialize store (registers builtin help/exit commands)
        self._command_store._initialize_sync()
        # Register default agent commands
        for command in get_commands():
            self._command_store.register_command(command)

        # Register additional provided commands
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

    @method_spawner  # type: ignore[misc]
    async def run_stream(
        self,
        *prompts: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Run agent with streaming output.

        This method delegates to _stream_events() which must be implemented by subclasses.
        Handles prompt conversion from various formats to UserContent.

        Args:
            *prompts: Input prompts (various formats supported)
            **kwargs: Additional arguments

        Yields:
            Stream events during execution
        """
        from agentpool.prompts.convert import convert_prompts

        # Convert prompts to standard UserContent format
        converted_prompts = await convert_prompts(prompts)

        async for event in self._stream_events(converted_prompts, **kwargs):
            yield event

    @abstractmethod
    def _stream_events(
        self,
        prompts: list[UserContent],
        *,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        input_provider: Any = None,
        message_history: Any = None,
        deps: Any = None,
        event_handlers: Any = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Agent-specific streaming implementation.

        Subclasses must implement this to provide their streaming logic.
        Prompts are pre-converted to UserContent format by run_stream().

        Args:
            prompts: Converted prompts in UserContent format
            message_id: Optional message ID
            conversation_id: Optional conversation ID
            parent_id: Optional parent message ID
            input_provider: Optional input provider
            message_history: Optional message history
            deps: Optional dependencies
            event_handlers: Optional event handlers
            wait_for_connections: Whether to wait for connected agents
            store_history: Whether to store in history

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

    def is_initializing(self) -> bool:
        """Check if agent is still initializing.

        Returns:
            True if deferred initialization is pending
        """
        return self._connect_pending

    async def ensure_initialized(self) -> None:
        """Wait for deferred initialization to complete.

        Subclasses that use deferred init should:
        1. Set `self._connect_pending = True` in `__aenter__`
        2. Override this method to do actual connection work
        3. Set `self._connect_pending = False` when done

        The base implementation is a no-op for agents without deferred init.
        """

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

    def get_mcp_server_info(self) -> dict[str, MCPServerStatus]:
        """Get information about configured MCP servers.

        Returns a dict mapping server names to their status info. Used by
        the OpenCode /mcp endpoint to display MCP servers in the UI.

        The default implementation checks external_providers on the tool manager.
        Subclasses may override to provide agent-specific MCP server info
        (e.g., ClaudeCodeAgent has its own MCP server handling).

        Returns:
            Dict mapping server name to MCPServerStatus
        """
        from agentpool.common_types import MCPServerStatus
        from agentpool.mcp_server.manager import MCPManager
        from agentpool.resource_providers import AggregatingResourceProvider
        from agentpool.resource_providers.mcp_provider import MCPResourceProvider

        def add_status(provider: MCPResourceProvider, result: dict[str, MCPServerStatus]) -> None:
            status_dict = provider.get_status()
            status_type = status_dict.get("status", "disabled")
            if status_type == "connected":
                result[provider.name] = MCPServerStatus(
                    name=provider.name, status="connected", server_type="stdio"
                )
            elif status_type == "failed":
                error = status_dict.get("error", "Unknown error")
                result[provider.name] = MCPServerStatus(
                    name=provider.name, status="error", error=error
                )
            else:
                result[provider.name] = MCPServerStatus(name=provider.name, status="disconnected")

        result: dict[str, MCPServerStatus] = {}
        try:
            for provider in self.tools.external_providers:
                if isinstance(provider, MCPResourceProvider):
                    add_status(provider, result)
                elif isinstance(provider, AggregatingResourceProvider):
                    for nested in provider.providers:
                        if isinstance(nested, MCPResourceProvider):
                            add_status(nested, result)
                elif isinstance(provider, MCPManager):
                    for mcp_provider in provider.get_mcp_providers():
                        add_status(mcp_provider, result)
        except Exception:  # noqa: BLE001
            pass

        return result

    @method_spawner
    async def run(
        self,
        *prompts: PromptCompatible | ChatMessage[Any],
        store_history: bool = True,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        message_history: MessageHistory | None = None,
        deps: TDeps | None = None,
        input_provider: InputProvider | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        wait_for_connections: bool | None = None,
    ) -> ChatMessage[TResult]:
        """Run agent with prompt and get response.

        This is the standard synchronous run method shared by all agent types.
        It collects all streaming events from run_stream() and returns the final message.

        Args:
            prompts: User query or instruction
            store_history: Whether the message exchange should be added to the
                            context window
            message_id: Optional message id for the returned message.
                        Automatically generated if not provided.
            conversation_id: Optional conversation id for the returned message.
            parent_id: Parent message id
            message_history: Optional MessageHistory object to
                             use instead of agent's own conversation
            deps: Optional dependencies for the agent
            input_provider: Optional input provider for the agent
            event_handlers: Optional event handlers for this run (overrides agent's handlers)
            wait_for_connections: Whether to wait for connected agents to complete

        Returns:
            ChatMessage containing response and run information

        Raises:
            RuntimeError: If no final message received from stream
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        from agentpool.agents.events import StreamCompleteEvent

        # Collect all events through run_stream
        final_message: ChatMessage[TResult] | None = None
        async for event in self.run_stream(
            *prompts,
            store_history=store_history,
            message_id=message_id,
            conversation_id=conversation_id,
            parent_id=parent_id,
            message_history=message_history,
            deps=deps,
            input_provider=input_provider,
            event_handlers=event_handlers,
            wait_for_connections=wait_for_connections,
        ):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        if final_message is None:
            msg = "No final message received from stream"
            raise RuntimeError(msg)

        return final_message

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
    async def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for this agent.

        Returns a list of mode categories that can be switched. Each category
        represents a group of mutually exclusive modes (e.g., permissions,
        models, behavior presets).

        Different agent types expose different modes:
        - Native Agent: permissions + model selection
        - ClaudeCodeAgent: permissions + model selection
        - ACPAgent: Passthrough from remote server
        - AGUIAgent: model selection (if applicable)

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
