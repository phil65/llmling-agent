"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, overload

from anyenv import MultiEventHandler, method_spawner
from anyenv.signals import BoundSignal, Signal
import anyio

from agentpool.agents.events import StreamCompleteEvent, resolve_event_handlers
from agentpool.agents.modes import ModeInfo
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage, MessageHistory, MessageNode
from agentpool.prompts.convert import convert_prompts
from agentpool.tools.manager import ToolManager
from agentpool.utils.inspection import call_with_context
from agentpool.utils.now import get_now


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime

    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import UserContent
    from slashed import BaseCommand, CommandStore
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp.schema import AvailableCommandsUpdate, ConfigOptionUpdate
    from agentpool.agents.agent import Agent
    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory, ModeInfo
    from agentpool.common_types import (
        AgentName,
        BuiltinEventHandlerType,
        IndividualEventHandler,
        MCPServerStatus,
        ProcessorCallback,
        PromptCompatible,
    )
    from agentpool.delegation import AgentPool, Team, TeamRun
    from agentpool.messaging import ChatMessage
    from agentpool.talk.stats import MessageStats
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode

    # Union type for state updates emitted via state_updated signal
    type StateUpdate = ModeInfo | ModelInfo | AvailableCommandsUpdate | ConfigOptionUpdate


logger = get_logger(__name__)


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

    Signals:
        - run_failed: Emitted when agent execution fails with error details
    """

    @dataclass(frozen=True)
    class RunFailedEvent:
        """Event emitted when agent execution fails."""

        agent_name: str
        """Name of the agent that failed."""
        message: str
        """Error description."""
        exception: Exception
        """The exception that caused the failure."""
        timestamp: Any = field(default_factory=get_now)  # datetime
        """When the failure occurred."""

    @dataclass(frozen=True)
    class AgentReset:
        """Emitted when agent is reset."""

        agent_name: AgentName
        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=get_now)

    agent_reset = Signal[AgentReset]()
    # Signal emitted when agent execution fails
    run_failed: Signal[RunFailedEvent] = Signal()

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
        self._infinite = False
        self._background_task: asyncio.Task[ChatMessage[Any]] | None = None

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

    @overload
    def __and__(  # if other doesnt define deps, we take the agents one
        self, other: ProcessorCallback[Any] | Team[TDeps] | Agent[TDeps, Any]
    ) -> Team[TDeps]: ...

    @overload
    def __and__(  # otherwise, we dont know and deps is Any
        self, other: ProcessorCallback[Any] | Team[Any] | Agent[Any, Any]
    ) -> Team[Any]: ...

    def __and__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> Team[Any]:
        """Create sequential team using & operator.

        Example:
            group = analyzer & planner & executor  # Create group of 3
            group = analyzer & existing_group  # Add to existing group
        """
        from agentpool.agents.agent import Agent
        from agentpool.delegation.team import Team

        match other:
            case Team():
                return Team([self, *other.nodes])
            case Callable():
                agent_2 = Agent.from_callback(other)
                agent_2.agent_pool = self.agent_pool
                return Team([self, agent_2])
            case MessageNode():
                return Team([self, other])
            case _:
                msg = f"Invalid agent type: {type(other)}"
                raise ValueError(msg)

    @overload
    def __or__(self, other: MessageNode[TDeps, Any]) -> TeamRun[TDeps, Any]: ...

    @overload
    def __or__[TOtherDeps](self, other: MessageNode[TOtherDeps, Any]) -> TeamRun[Any, Any]: ...

    @overload
    def __or__(self, other: ProcessorCallback[Any]) -> TeamRun[Any, Any]: ...

    def __or__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> TeamRun[Any, Any]:
        # Create new execution with sequential mode (for piping)
        from agentpool import TeamRun
        from agentpool.agents.agent import Agent

        if callable(other):
            other = Agent.from_callback(other)
            other.agent_pool = self.agent_pool

        return TeamRun([self, other])

    @property
    def command_store(self) -> CommandStore:
        """Get the command store for slash commands."""
        return self._command_store

    async def reset(self) -> None:
        """Reset agent state (conversation history and tool states)."""
        old_tools = await self.tools.list_tools()
        await self.conversation.clear()
        await self.tools.reset_states()
        new_tools = await self.tools.list_tools()

        event = self.AgentReset(
            agent_name=self.name,
            previous_tools=old_tools,
            new_tools=new_tools,
        )
        await self.agent_reset.emit(event)

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

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
        store_history: bool = True,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[ChatMessage[TResult]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially
            store_history: Whether to store in conversation history
            wait_for_connections: Whether to wait for connected agents

        Yields:
            Response messages in sequence

        Example:
            questions = [
                ["What is your name?"],
                ["How old are you?", image1],
                ["Describe this image", image2],
            ]
            async for response in agent.run_iter(*questions):
                print(response.content)
        """
        for prompts in prompt_groups:
            response = await self.run(
                *prompts,
                store_history=store_history,
                wait_for_connections=wait_for_connections,
            )
            yield response  # pyright: ignore

    async def run_in_background(
        self,
        *prompt: PromptCompatible,
        max_count: int | None = None,
        interval: float = 1.0,
        **kwargs: Any,
    ) -> asyncio.Task[ChatMessage[TResult] | None]:
        """Run agent continuously in background with prompt or dynamic prompt function.

        Args:
            prompt: Static prompt or function that generates prompts
            max_count: Maximum number of runs (None = infinite)
            interval: Seconds between runs
            **kwargs: Arguments passed to run()
        """
        self._infinite = max_count is None

        async def _continuous() -> ChatMessage[Any]:
            count = 0
            self.log.debug("Starting continuous run", max_count=max_count, interval=interval)
            latest = None
            while (max_count is None or count < max_count) and not self._cancelled:
                try:
                    agent_ctx = self.get_context()
                    current_prompts = [
                        call_with_context(p, agent_ctx, **kwargs) if callable(p) else p
                        for p in prompt
                    ]
                    self.log.debug("Generated prompt", iteration=count)
                    latest = await self.run(current_prompts, **kwargs)
                    self.log.debug("Run continuous result", iteration=count)

                    count += 1
                    await anyio.sleep(interval)
                except asyncio.CancelledError:
                    self.log.debug("Continuous run cancelled")
                    break
                except Exception:
                    # Check if we were cancelled (may surface as other exceptions)
                    if self._cancelled:
                        self.log.debug("Continuous run cancelled via flag")
                        break
                    count += 1
                    self.log.exception("Background run failed")
                    await anyio.sleep(interval)
            self.log.debug("Continuous run completed", iterations=count)
            return latest  # type: ignore[return-value]

        await self.stop()  # Cancel any existing background task
        self._cancelled = False  # Reset cancellation flag for new run
        task = asyncio.create_task(_continuous(), name=f"background_{self.name}")
        self.log.debug("Started background task", task_name=task.get_name())
        self._background_task = task
        return task

    async def stop(self) -> None:
        """Stop continuous execution if running."""
        self._cancelled = True  # Signal cancellation via flag
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            with suppress(asyncio.CancelledError):  # Expected when we cancel the task
                await self._background_task
            self._background_task = None

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self.task_manager._pending_tasks or self._background_task)

    async def wait(self) -> ChatMessage[TResult]:
        """Wait for background execution to complete."""
        if not self._background_task:
            msg = "No background task running"
            raise RuntimeError(msg)
        if self._infinite:
            msg = "Cannot wait on infinite execution"
            raise RuntimeError(msg)
        try:
            return await self._background_task
        finally:
            self._background_task = None

    @method_spawner
    async def run_stream(
        self,
        *prompts: PromptCompatible,
        store_history: bool = True,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        message_history: MessageHistory | None = None,
        input_provider: InputProvider | None = None,
        wait_for_connections: bool | None = None,
        deps: TDeps | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Run agent with streaming output.

        This method delegates to _stream_events() which must be implemented by subclasses.
        Handles prompt conversion from various formats to UserContent.

        Args:
            *prompts: Input prompts (various formats supported)
            store_history: Whether to store in history
            message_id: Optional message ID
            conversation_id: Optional conversation ID
            parent_id: Optional parent message ID
            message_history: Optional message history
            input_provider: Optional input provider
            wait_for_connections: Whether to wait for connected agents
            deps: Optional dependencies
            event_handlers: Optional event handlers

        Yields:
            Stream events during execution
        """
        from agentpool.messaging import ChatMessage
        from agentpool.utils.identifiers import generate_session_id

        # Convert prompts to standard UserContent format
        converted_prompts = await convert_prompts(prompts)
        # Get message history (either passed or agent's own)
        conversation = message_history if message_history is not None else self.conversation
        # Determine effective parent_id (from param or last message in history)
        effective_parent_id = parent_id if parent_id else conversation.get_last_message_id()
        # Initialize or adopt conversation_id
        if self.conversation_id is None:
            if conversation_id:
                # Adopt conversation_id (from agent chain or external session like ACP)
                self.conversation_id = conversation_id
            else:
                # Generate new conversation_id
                self.conversation_id = generate_session_id()
            # Always log conversation with initial prompt for title generation
            # StorageManager handles idempotent behavior (skip if already logged)
            # Use last prompt to avoid staged content (staged is prepended, user prompt is last)
            user_prompts = [
                str(p) for p in prompts if isinstance(p, str)
            ]  # Filter to text prompts only
            initial_prompt = user_prompts[-1] if user_prompts else None
            await self.log_conversation(initial_prompt)
        elif conversation_id and self.conversation_id != conversation_id:
            # Adopt passed conversation_id (for routing chains)
            self.conversation_id = conversation_id

        user_msg = ChatMessage.user_prompt(
            message=converted_prompts,
            parent_id=effective_parent_id,
            conversation_id=self.conversation_id,
        )

        # Stream events from implementation
        final_message = None
        self._current_stream_task = asyncio.current_task()
        try:
            async for event in self._stream_events(
                converted_prompts,
                user_msg=user_msg,
                effective_parent_id=effective_parent_id,
                store_history=store_history,
                message_id=message_id,
                conversation_id=conversation_id,
                parent_id=parent_id,
                message_history=message_history,
                input_provider=input_provider,
                wait_for_connections=wait_for_connections,
                deps=deps,
                event_handlers=event_handlers,
            ):
                yield event
                # Capture final message from StreamCompleteEvent
                if isinstance(event, StreamCompleteEvent):
                    final_message = event.message
        except Exception as e:
            self.log.exception("Agent stream failed")
            failed_event = BaseAgent.RunFailedEvent(
                agent_name=self.name,
                message="Agent stream failed",
                exception=e,
            )
            await self.run_failed.emit(failed_event)
            raise
        finally:
            self._current_stream_task = None

        # Post-processing after stream completes
        if final_message is not None:
            # Emit signal (always - for event handlers)
            await self.message_sent.emit(final_message)
            # Conditional persistence based on store_history
            # TODO: Verify store_history semantics across all use cases:
            #   - Should subagent tool calls set store_history=False?
            #   - Should forked/ephemeral runs always skip persistence?
            #   - Should signals still fire when store_history=False?
            #   Current behavior: store_history controls both DB logging AND conversation context
            if store_history:
                # Log to persistent storage and add to conversation context
                await self.log_message(final_message)
                conversation.add_chat_messages([user_msg, final_message])
            # Route to connected agents (always - they decide what to do with it)
            await self.connections.route_message(final_message, wait=wait_for_connections)

    @abstractmethod
    def _stream_events(
        self,
        prompts: list[UserContent],
        *,
        user_msg: Any,  # ChatMessage but imported in run_stream
        effective_parent_id: str | None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
        deps: TDeps | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Agent-specific streaming implementation.

        Subclasses must implement this to provide their streaming logic.
        Prompts are pre-converted to UserContent format by run_stream().

        Args:
            prompts: Converted prompts in UserContent format
            user_msg: Pre-created user ChatMessage (from base class)
            effective_parent_id: Resolved parent message ID for threading
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
