"""ACP Agent - MessageNode wrapping an external ACP subprocess.

This module provides an agent implementation that communicates with external
ACP (Agent Client Protocol) servers via stdio, enabling integration of any
ACP-compatible agent into the agentpool pool.

The ACPAgent class acts as an ACP client, spawning an ACP server subprocess
and communicating with it via JSON-RPC over stdio. This allows:
- Integration of external ACP-compatible agents (like claude-code-acp)
- Composition with native agents via connections, teams, etc.
- Full ACP protocol support including file operations and terminals

Example:
    ```python
    from agentpool.models.acp_agents import ACPAgentConfig

    config = ACPAgentConfig(
        command="claude-code-acp",
        name="claude_coder",
        cwd="/path/to/project",
    )
    async with ACPAgent(config=config) as agent:
        result = await agent.run("Write a hello world program")
        print(result.content)
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self
import uuid

import anyio
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from agentpool.agents.acp_agent.acp_converters import event_to_part
from agentpool.agents.acp_agent.session_state import ACPSessionState
from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.events import (
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolResultMetadataEvent,
)
from agentpool.agents.events.processors import FileTracker
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.models.acp_agents import ACPAgentConfig
from agentpool.utils.streams import merge_queue_into_iterator
from agentpool.utils.subprocess_utils import SubprocessError, run_with_process_monitor
from agentpool.utils.token_breakdown import calculate_usage_from_parts


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
    from types import TracebackType

    from anyenv import MultiEventHandler
    from anyio.abc import Process
    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import ThinkingPart, ToolCallPart, UserContent
    from slashed import BaseCommand
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp.agent.protocol import Agent as ACPAgentProtocol
    from acp.client.connection import ClientSideConnection
    from acp.client.protocol import Client
    from acp.schema import Implementation, RequestPermissionRequest, RequestPermissionResponse
    from acp.schema.capabilities import AgentCapabilities
    from acp.schema.mcp import McpServer
    from agentpool.agents import AgentContext
    from agentpool.agents.acp_agent.client_handler import ACPClientHandler
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import BuiltinEventHandlerType, IndividualEventHandler
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageHistory
    from agentpool.models.acp_agents import BaseACPAgentConfig
    from agentpool.sessions import SessionData
    from agentpool.ui.base import InputProvider
    from agentpool_config.nodes import ToolConfirmationMode

logger = get_logger(__name__)


def get_updated_at(date_str: str | None) -> datetime:
    from agentpool.utils.now import get_now

    updated_at = get_now()
    if date_str:
        with contextlib.suppress(ValueError, AttributeError):
            updated_at = datetime.fromisoformat(date_str)
    return updated_at


class ACPAgent[TDeps = None](BaseAgent[TDeps, str]):
    """MessageNode that wraps an external ACP agent subprocess.

    This allows integrating any ACP-compatible agent into the agentpool
    pool, enabling composition with native agents via connections, teams, etc.

    The agent manages:
    - Subprocess lifecycle (spawn on enter, terminate on exit)
    - ACP protocol initialization and session creation
    - Prompt execution with session update collection
    - Client-side operations (filesystem, terminals, permissions)

    Supports both blocking `run()` and streaming `run_iter()` execution modes.

    Example:
        ```python
        # From config
        config = ClaudeACPAgentConfig(cwd="/project")
        agent = ACPAgent(config=config, agent_pool=pool)

        # From kwargs
        agent = ACPAgent(command="claude-code-acp", cwd="/project")
        ```
    """

    AGENT_TYPE: ClassVar = "acp"

    def __init__(
        self,
        *,
        config: BaseACPAgentConfig | None = None,
        command: str | None = None,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "always",
        commands: Sequence[BaseCommand] | None = None,
        hooks: AgentHooks | None = None,
        session_id: str | None = None,
    ) -> None:
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge

        # Build config from kwargs if not provided
        if config is None:
            if command is None:
                raise ValueError("Either config or command must be provided")
            config = ACPAgentConfig(
                name=name,
                description=description,
                display_name=display_name,
                command=command,
                args=args or [],
                cwd=cwd,
                env=env_vars or {},
                allow_file_operations=allow_file_operations,
                allow_terminal=allow_terminal,
                requires_tool_confirmation=tool_confirmation_mode,
            )

        super().__init__(
            name=name or config.name or config.get_command(),
            description=description or config.description,
            display_name=display_name or config.display_name,
            mcp_servers=config.mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs or list(config.triggers),
            env=config.get_execution_environment(),
            input_provider=input_provider,
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
            commands=commands,
            hooks=hooks,
        )

        # ACP-specific state
        self.acp_permission_callback: (
            Callable[[RequestPermissionRequest], Awaitable[RequestPermissionResponse]] | None
        ) = None
        self.config = config
        self._process: Process | None = None
        self._connection: ClientSideConnection | None = None
        self._client_handler: ACPClientHandler | None = None
        self._agent_info: Implementation | None = None
        self._caps: AgentCapabilities | None = None
        self._session_id: str | None = session_id  # Session ID to load or from new_session
        self._state: ACPSessionState | None = None
        self.deps_type = type(None)
        self._extra_mcp_servers: list[McpServer] = []
        self._sessions_cache: list[SessionData] | None = None  # Cache for list_sessions results
        # Create bridge (not started yet) - will be started in _setup_toolsets if needed
        self._tool_bridge = ToolManagerBridge(node=self, server_name=f"agentpool-{self.name}-tools")
        # Client execution environment (for subprocess requests) - falls back to env
        self._client_env: ExecutionEnvironment | None = config.get_client_execution_environment()
        # Track the prompt task for cancellation
        self._prompt_task: asyncio.Task[Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: BaseACPAgentConfig,
        *,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
    ) -> Self:
        """Create an ACPAgent from a config object."""
        # Merge config-level handlers with provided handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]
        return cls(
            config=config,
            event_handlers=merged_handlers or None,
            input_provider=input_provider,
            agent_pool=agent_pool,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
        )

    @property
    def client_env(self) -> ExecutionEnvironment:
        """Execution environment for handling subprocess requests.

        This is used by ACPClientHandler for file/terminal operations requested
        by the subprocess. Falls back to the agent's main env if not explicitly set.

        Use cases:
        - Default (None): Subprocess requests use same env as toolsets
        - Explicit: Subprocess operates in a different environment than toolsets
        """
        return self._client_env if self._client_env is not None else self.env

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
    ) -> AgentContext:
        """Create a new context for this agent.

        Args:
            data: Optional custom data to attach to the context
            input_provider: Optional input provider override
        """
        from agentpool.agents.context import AgentContext
        from agentpool.models.manifest import AgentsManifest

        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return AgentContext(
            node=self,
            pool=self.agent_pool,
            config=self.config,
            definition=defn,
            input_provider=input_provider or self._input_provider,
            data=data,
            model_name=self.model_name,
        )

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets from config and start bridge if needed."""
        if not self.config.tools:
            return
        # Create providers from tool configs and add to tool manager
        for provider in self.config.get_tool_providers():
            self.tools.add_provider(provider)
        await self._tool_bridge.start()  # Start bridge to expose tools via MCP
        mcp_config = self._tool_bridge.get_mcp_server_config()
        self._extra_mcp_servers.append(mcp_config)

    async def __aenter__(self) -> Self:
        """Start subprocess and initialize ACP connection."""
        await super().__aenter__()
        await self._setup_toolsets()  # Setup toolsets before session creation
        process = await self._start_process()
        try:
            await run_with_process_monitor(process, self._initialize, context="ACP initialization")
            # Load existing session or create new one
            if self._session_id:
                session_to_load = self._session_id
                self._session_id = None  # Clear so load_session can set it
                result = await run_with_process_monitor(
                    process,
                    lambda: self.load_session(session_to_load),
                    context="ACP session load",
                )
                if result is None:
                    # Fall back to creating a new session if load fails
                    self.log.warning(
                        "Failed to load session, creating new one",
                        session_id=session_to_load,
                    )
                    await run_with_process_monitor(
                        process, self._create_session, context="ACP session creation"
                    )
            else:
                await run_with_process_monitor(
                    process, self._create_session, context="ACP session creation"
                )
        except SubprocessError as e:
            raise RuntimeError(str(e)) from e
        await anyio.sleep(0.3)  # Small delay to let subprocess fully initialize
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up subprocess and connection."""
        await self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _start_process(self) -> Process:
        """Start the ACP server subprocess.

        Returns:
            The started Process instance
        """
        prompt_manager = self.agent_pool.prompt_manager if self.agent_pool else None
        args = await self.config.get_args(prompt_manager)
        cmd = [self.config.get_command(), *args]
        self.log.info("Starting ACP subprocess", command=cmd)
        env = {**os.environ, **self.config.env}
        cwd = str(self.config.cwd) if self.config.cwd else None
        self._process = await anyio.open_process(cmd, env=env, cwd=cwd)
        if not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Failed to create subprocess pipes")
        return self._process

    async def _initialize(self) -> None:
        """Initialize the ACP connection."""
        from acp.client.connection import ClientSideConnection
        from agentpool.agents.acp_agent.client_handler import ACPClientHandler

        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Process not started")

        self._state = ACPSessionState(session_id="")
        self._client_handler = ACPClientHandler(self, self._state, self._input_provider)

        def client_factory(agent: ACPAgentProtocol) -> Client:
            return self._client_handler  # type: ignore[return-value]

        self._connection = ClientSideConnection(
            to_client=client_factory,
            input_stream=self._process.stdin,
            output_stream=self._process.stdout,
        )
        init_request = self.config.create_initialize_request()
        init_response = await self._connection.initialize(init_request)
        self._agent_info = init_response.agent_info
        self._caps = init_response.agent_capabilities
        self.log.info("ACP connection initialized", agent_info=self._agent_info)

    async def _create_session(self) -> None:
        """Create a new ACP session with configured MCP servers."""
        from acp.schema import NewSessionRequest
        from agentpool.agents.acp_agent.acp_converters import mcp_config_to_acp
        from agentpool.agents.acp_agent.helpers import filter_servers_by_capabilities

        if not self._connection:
            raise RuntimeError("Connection not initialized")

        # Collect all MCP servers (config + extra)
        all_servers = self._extra_mcp_servers[:]
        # Add servers from config (converted to ACP format)
        if config_servers := self.config.get_mcp_servers():
            all_servers.extend([mcp_config_to_acp(config) for config in config_servers])
        mcp_servers = filter_servers_by_capabilities(all_servers, self._caps, logger=self.log)
        cwd = self.config.cwd or str(Path.cwd())
        session_request = NewSessionRequest(cwd=cwd, mcp_servers=mcp_servers)
        response = await self._connection.new_session(session_request)
        self._session_id = response.session_id
        if self._state:
            self._state.session_id = self._session_id
            # Store config_options if available (newer ACP protocol)
            if response.config_options:
                self._state.config_options = list(response.config_options)
            # Legacy: Store models and modes for backward compatibility
            if response.models:  # Store full model info from session response
                self._state.models = response.models
                self._state.current_model_id = response.models.current_model_id
            self._state.modes = response.modes
        model = self._state.current_model_id if self._state else None
        self.log.info("ACP session created", session_id=self._session_id, model=model)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Only stop bridge if it was started (has _mcp set)
        if self._tool_bridge._mcp is not None:
            await self._tool_bridge.stop()
        self._extra_mcp_servers.clear()
        if self._client_handler:
            await self._client_handler.cleanup()
            self._client_handler = None
        if self._connection:
            try:
                await self._connection.close()
            except Exception:
                self.log.exception("Error closing ACP connection")
            self._connection = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception:
                self.log.exception("Error terminating ACP process")
            self._process = None

    async def _stream_events(  # noqa: PLR0915
        self,
        prompts: list[UserContent],
        *,
        user_msg: ChatMessage[Any],
        message_history: MessageHistory,
        effective_parent_id: str | None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        event_handlers: MultiEventHandler[IndividualEventHandler],
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:

        from acp.schema import ForkSessionRequest, PromptRequest
        from agentpool.agents.acp_agent.acp_converters import (
            convert_to_acp_content,
            to_finish_reason,
        )

        # Update input provider if provided
        if input_provider is not None and self._client_handler:
            self._client_handler._input_provider = input_provider
        if not self._connection or not self._session_id or not self._state:
            raise RuntimeError("Agent not initialized - use async context manager")

        run_id = str(uuid.uuid4())
        self._state.clear()  # Reset state
        # Track messages in pydantic-ai format: ModelRequest -> ModelResponse -> ...
        # This mirrors pydantic-ai's new_messages() which includes the initial user request.
        model_messages: list[ModelResponse | ModelRequest] = []
        # Start with the user's request (same as pydantic-ai's new_messages())
        initial_request = ModelRequest(parts=[UserPromptPart(content=prompts)])
        model_messages.append(initial_request)
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []  # For final content string
        file_tracker = FileTracker()  # Track files modified by tool calls
        assert self.conversation_id is not None  # Initialized by BaseAgent.run_stream()
        run_started = RunStartedEvent(
            thread_id=self.conversation_id,
            run_id=run_id,
            agent_name=self.name,
        )
        await event_handlers(None, run_started)
        yield run_started
        final_blocks = convert_to_acp_content(prompts)
        # Handle ephemeral execution (fork session if store_history=False)
        session_id = self._session_id
        if not store_history and self._session_id:
            # Fork the current session to execute without affecting main history
            cwd = self.config.cwd or str(Path.cwd())
            fork_request = ForkSessionRequest(session_id=self._session_id, cwd=cwd)
            fork_response = await self._connection.fork_session(fork_request)
            # Use the forked session ID for this prompt
            session_id = fork_response.session_id
            self.log.debug("Forked session", parent=self._session_id, fork=session_id)
        prompt_request = PromptRequest(session_id=session_id, prompt=final_blocks)
        self.log.debug("Starting streaming prompt", num_blocks=len(final_blocks))
        # Run prompt in background
        prompt_task = asyncio.create_task(self._connection.prompt(prompt_request))
        self._prompt_task = prompt_task

        # Create async generator that polls ACP events
        async def poll_acp_events() -> AsyncIterator[RichAgentStreamEvent[str]]:
            """Poll raw updates from ACP state, convert to events, until prompt completes."""
            from agentpool.agents.acp_agent.acp_converters import acp_to_native_event

            assert self._state
            while not prompt_task.done():
                if self._client_handler:
                    try:
                        await self._client_handler._update_event.wait_with_timeout(0.05)
                        self._client_handler._update_event.clear()
                    except TimeoutError:
                        pass
                # Pop and convert pending raw updates
                while (update := self._state.pop_update()) is not None:
                    if native_event := acp_to_native_event(update):
                        yield native_event
            # Pop remaining updates after prompt completes
            while (update := self._state.pop_update()) is not None:
                if native_event := acp_to_native_event(update):
                    yield native_event

        # Accumulate metadata events by tool_call_id (workaround for MCP stripping _meta)
        tool_metadata: dict[str, dict[str, Any]] = {}
        # Merge ACP events with custom events from queue
        # Set deps/input_provider on tool bridge (ContextVar doesn't work - separate task)
        try:
            async with (
                self._tool_bridge.set_run_context(deps, input_provider, prompt=prompts),
                merge_queue_into_iterator(poll_acp_events(), self._event_queue) as merged_events,
            ):
                async for event in file_tracker(merged_events):
                    # Capture metadata events for correlation with tool results
                    if isinstance(event, ToolResultMetadataEvent):
                        tool_metadata[event.tool_call_id] = event.metadata
                        # Don't yield metadata events - they're internal correlation only
                        continue
                    # Check for cancellation
                    if self._cancelled:
                        self.log.info("Stream cancelled by user")
                        break
                    # Inject metadata into ToolCallCompleteEvent
                    # (converted from completed ToolCallProgress)
                    if isinstance(event, ToolCallCompleteEvent):
                        # Enrich with agent name and metadata from our accumulator
                        enriched_event = event
                        if not enriched_event.agent_name:
                            enriched_event = replace(enriched_event, agent_name=self.name)
                        if (
                            enriched_event.metadata is None
                            and enriched_event.tool_call_id in tool_metadata
                        ):
                            enriched_event = replace(
                                enriched_event, metadata=tool_metadata[enriched_event.tool_call_id]
                            )
                        event = enriched_event  # noqa: PLW2901
                    # Extract content from events and build parts in arrival order
                    part = event_to_part(event)
                    if isinstance(part, TextPart):
                        text_chunks.append(part.content)
                    if part:
                        current_response_parts.append(part)
                    await event_handlers(None, event)
                    yield event
        except asyncio.CancelledError:
            self.log.info("Stream cancelled via task cancellation")
            self._cancelled = True

        # Handle cancellation - emit partial message
        if self._cancelled:
            message = ChatMessage[str](
                content="".join(text_chunks),
                role="assistant",
                name=self.name,
                message_id=message_id or str(uuid.uuid4()),
                conversation_id=self.conversation_id,
                parent_id=user_msg.message_id,
                model_name=self.model_name,
                messages=model_messages,
                metadata=file_tracker.get_metadata(),
                finish_reason="stop",
            )
            complete_event = StreamCompleteEvent(message=message)
            await event_handlers(None, complete_event)
            yield complete_event
            self._prompt_task = None
            return

        # Ensure we catch any exceptions from the prompt task
        response = await prompt_task
        finish_reason = to_finish_reason(response.stop_reason)
        # Flush response parts to model_messages
        if current_response_parts:
            model_messages.append(
                ModelResponse(
                    parts=current_response_parts,
                    finish_reason=finish_reason,
                    model_name=self.model_name,
                    provider_name=self.config.type,
                )
            )

        text_content = "".join(text_chunks)
        # Calculate approximate token usage from what we can observe
        usage, cost_info = await calculate_usage_from_parts(
            input_parts=prompts,
            response_parts=current_response_parts,
            text_content=text_content,
            model_name=self.model_name,
            provider=self.config.type,
        )

        message = ChatMessage[str](
            content=text_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            parent_id=user_msg.message_id,
            model_name=self.model_name,
            messages=model_messages,
            metadata=file_tracker.get_metadata(),
            finish_reason=finish_reason,
            usage=usage,
            cost_info=cost_info,
        )
        complete_event = StreamCompleteEvent(message=message)
        await event_handlers(None, complete_event)
        yield complete_event  # Emit final StreamCompleteEvent - post-processing handled by base

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        return model_id if self._state and (model_id := self._state.current_model_id) else None

    async def set_model(self, model: str) -> None:
        """Update the model for the current session via ACP protocol."""
        await self._set_mode(model, "model")

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set the tool confirmation mode for this agent.

        For ACPAgent, this sends a set_session_mode request to the remote ACP server
        to change its mode. The mode is also stored locally for the client handler.

        Note: "per_tool" behaves like "always" since we don't have per-tool metadata
        from the ACP server.

        Args:
            mode: Tool confirmation mode
        """
        from acp.schema import SetSessionModeRequest
        from agentpool_server.acp_server.converters import confirmation_mode_to_mode_id

        self.tool_confirmation_mode = mode
        if self._client_handler:  # Update client handler if it exists
            self._client_handler.tool_confirmation_mode = mode

        # Forward mode change to remote ACP server if connected
        if self._connection and self._session_id:
            mode_id = confirmation_mode_to_mode_id(mode)
            request = SetSessionModeRequest(session_id=self._session_id, mode_id=mode_id)
            try:
                await self._connection.set_session_mode(request)
                msg = "Forwarded mode change to remote ACP server"
                self.log.info(msg, mode=mode, mode_id=mode_id)
            except Exception:
                self.log.exception("Failed to forward mode change to remote ACP server")
        else:
            self.log.info("Tool confirmation mode changed (local only)", mode=mode)

    async def _interrupt(self) -> None:
        """Send CancelNotification to remote ACP server and cancel local tasks."""
        from acp.schema import CancelNotification

        # Send cancel notification to the remote ACP server
        if self._connection and self._session_id:
            try:
                cancel_notification = CancelNotification(session_id=self._session_id)
                await self._connection.cancel(cancel_notification)
                self.log.info("Sent cancel notification to ACP server")
            except Exception:
                self.log.exception("Failed to send cancel notification to ACP server")

        # Cancel the local prompt task
        if self._prompt_task and not self._prompt_task.done():
            self._prompt_task.cancel()
            self.log.info("Cancelled prompt task")
        # Also cancel current stream task (from base class)
        if self._current_stream_task and not self._current_stream_task.done():
            self._current_stream_task.cancel()

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models from the ACP session state.

        Converts ACP ModelInfo to tokonomics ModelInfo format.

        Returns:
            List of tokonomics ModelInfo, or None if not available
        """
        from tokonomics.model_discovery.model_info import ModelInfo

        if not self._state or not self._state.models:
            return None
        # Convert ACP ModelInfo to tokonomics ModelInfo
        return [
            ModelInfo(id=m.model_id, name=m.name, description=m.description)
            for m in self._state.models.available_models
        ]

    async def get_modes(self) -> list[ModeCategory]:
        """Get available modes from the ACP session state.

        Passthrough from remote ACP server's mode and model state.
        Prefers new config_options format, falls back to legacy modes/models.

        Returns:
            List of ModeCategory from remote server, empty if not available
        """
        from agentpool.agents.acp_agent.acp_converters import get_modes

        if not self._state:
            return []

        # Prefer new SessionConfigOption format if available
        return get_modes(
            self._state.config_options,
            available_modes=self._state.modes,
            available_models=self._state.models,
        )

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """Forward mode change to remote ACP server."""
        from acp.schema import (
            SetSessionConfigOptionRequest,
            SetSessionModelRequest,
            SetSessionModeRequest,
        )
        from agentpool.agents.modes import ConfigOptionChanged

        if not self._connection or not self._session_id or not self._state:
            raise RuntimeError("Not connected to ACP server")
        # Validate mode is available
        available_modes = await self.get_modes()
        matching_category = next((c for c in available_modes if c.id == category_id), None)
        if matching_category:
            valid_ids = {m.id for m in matching_category.available_modes}
            if mode_id not in valid_ids:
                raise ValueError(f"Unknown {category_id}: {mode_id}. Available: {valid_ids}")
        else:
            available_cats = {c.id for c in available_modes}
            raise ValueError(f"Unknown category: {category_id}. Available: {available_cats}")
        # Prefer new config_options API if available
        if self._state.config_options:
            assert category_id
            config_request = SetSessionConfigOptionRequest(
                session_id=self._session_id,
                config_id=category_id,
                value=mode_id,
            )
            response = await self._connection.set_session_config_option(config_request)
            # Update local state from response
            if response.config_options:
                self._state.config_options = list(response.config_options)
        # Legacy: Use old set_session_mode/set_session_model APIs
        elif category_id == "mode":
            mode_request = SetSessionModeRequest(session_id=self._session_id, mode_id=mode_id)
            await self._connection.set_session_mode(mode_request)
            # Update local state
            if self._state.modes:
                self._state.modes.current_mode_id = mode_id
        elif category_id == "model":
            # Legacy: Use set_session_model API
            request = SetSessionModelRequest(session_id=self._session_id, model_id=mode_id)
            if await self._connection.set_session_model(request):
                self._state.current_model_id = mode_id
                self.log.info("Model changed via legacy set_session_model")
            else:
                raise RuntimeError("Remote ACP agent does not support model changes.")
        else:
            raise ValueError(f"Unknown category: {category_id}. Available: mode, model")
        self.log.info("Config option changed", config_id=category_id, value=mode_id)
        change = ConfigOptionChanged(config_id=category_id, value_id=mode_id)
        await self.state_updated.emit(change)

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[SessionData]:
        """List sessions from the remote ACP server."""
        from acp.schema import ListSessionsRequest
        from agentpool.sessions.models import SessionData

        if not self._connection:
            raise RuntimeError("Not connected to ACP server")
        # Pass cwd filter to ACP server request
        request = ListSessionsRequest(cwd=cwd)
        try:
            response = await self._connection.list_sessions(request)
        except Exception:
            self.log.exception("Failed to list sessions from ACP server")
            return []
        else:
            # Convert ACP SessionInfo to agentpool SessionData
            result: list[SessionData] = []
            for acp_session in response.sessions:
                updated_at = get_updated_at(acp_session.updated_at)
                meta = acp_session.field_meta or {}
                # Extract created_at from _meta if available, otherwise use updated_at
                created_at = updated_at
                if meta_created := meta.get("created_at"):
                    created_at = get_updated_at(meta_created)
                session_data = SessionData(
                    session_id=acp_session.session_id,
                    agent_name=self.name,
                    conversation_id=acp_session.session_id,
                    cwd=acp_session.cwd,
                    created_at=created_at,
                    last_active=updated_at,
                    # Extract optional fields from _meta
                    pool_id=meta.get("pool_id"),
                    project_id=meta.get("project_id"),
                    parent_id=meta.get("parent_id"),
                    version=meta.get("version", "1"),
                    metadata={"title": acp_session.title} if acp_session.title else {},
                )
                result.append(session_data)
            # Update cache with full results (before applying limit)
            self._sessions_cache = result
            # Apply limit (ACP doesn't support limit in request yet)
            if limit is not None:
                result = result[:limit]
            return result

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load and restore a session from the remote ACP server.

        Per ACP protocol spec, the server will:
        1. Replay the entire conversation history via session/update notifications
        2. Return the session/load response AFTER all updates have been sent

        This method collects those replayed updates and converts them to ChatMessage
        objects to populate the agent's conversation history.
        """
        from acp.schema import LoadSessionRequest
        from agentpool.agents.acp_agent.acp_converters import (
            acp_notifications_to_messages,
            mcp_config_to_acp,
        )
        from agentpool.agents.acp_agent.helpers import filter_servers_by_capabilities
        from agentpool.sessions.models import SessionData
        from agentpool.utils.now import get_now

        if not self._connection:
            self.log.error("Cannot load session: not connected to ACP server")
            return None

        if not self._state:
            self.log.error("Cannot load session: state not initialized")
            return None

        try:
            # Collect all MCP servers (config + extra) for the load request
            all_servers = self._extra_mcp_servers[:]
            if config_servers := self.config.get_mcp_servers():
                all_servers.extend([mcp_config_to_acp(config) for config in config_servers])
            mcp_servers = filter_servers_by_capabilities(all_servers, self._caps, logger=self.log)
            cwd = self.config.cwd or str(Path.cwd())
            load_request = LoadSessionRequest(
                session_id=session_id, cwd=cwd, mcp_servers=mcp_servers or None
            )

            # Start collecting raw updates for conversation reconstruction
            self._state.start_load()

            # Load session on the remote server.
            # Per ACP spec, the server replays all conversation history via session/update
            # notifications BEFORE returning this response.
            response = await self._connection.load_session(load_request)

            # Collect the raw updates that were received during load
            raw_updates = self._state.finish_load()

            # Update local session ID and state
            self._session_id = session_id
            self._state.session_id = session_id

            # Update config_options if available
            if response.config_options:
                self._state.config_options = list(response.config_options)
            # Legacy: Update models and modes for backward compatibility
            if response.models:
                self._state.models = response.models
                self._state.current_model_id = response.models.current_model_id
            if response.modes:
                self._state.modes = response.modes

            # Convert replayed ACP updates to ChatMessage objects
            if raw_updates:
                chat_messages = acp_notifications_to_messages(
                    raw_updates,
                    conversation_id=session_id,
                    agent_name=self.name,
                    model_name=self.model_name,
                )
                # Populate the agent's conversation with reconstructed history
                self.conversation.chat_messages.clear()
                self.conversation.chat_messages.extend(chat_messages)
                self.log.info(
                    "Restored conversation history",
                    session_id=session_id,
                    message_count=len(chat_messages),
                    update_count=len(raw_updates),
                )
            else:
                self.log.debug("No conversation history to restore", session_id=session_id)

            self.log.info("Session loaded from ACP server", session_id=session_id)

            # Try to get session metadata from cache first
            def find_in_cache(sid: str) -> SessionData | None:
                if self._sessions_cache is None:
                    return None
                return next((s for s in self._sessions_cache if s.session_id == sid), None)

            # Check cache first
            if session_info := find_in_cache(session_id):
                return session_info

            # Not in cache, refresh and try again
            try:
                await self.list_sessions()  # This populates cache
                if session_info := find_in_cache(session_id):
                    return session_info
            except Exception:  # noqa: BLE001
                self.log.debug("Could not fetch session metadata", session_id=session_id)

            # Fallback: Return minimal SessionData
            return SessionData(
                session_id=session_id,
                agent_name=self.name,
                conversation_id=session_id,
                cwd=cwd,
                last_active=get_now(),
                created_at=get_now(),
            )

        except Exception:
            # Ensure we clean up load state on error
            if self._state:
                self._state.finish_load()
            self.log.exception("Failed to load session from ACP server", session_id=session_id)
            return None


if __name__ == "__main__":
    from agentpool.models.acp_agents import ACPAgentConfig

    async def main() -> None:
        """Demo: Basic call to an ACP agent."""
        config = ACPAgentConfig(command="uv", args=["run", "agentpool", "serve-acp"])
        async with ACPAgent(config=config, event_handlers=["detailed"]) as agent:
            print("Response (streaming): ", end="", flush=True)
            async for chunk in agent.run_stream("Say hello briefly."):
                print(chunk, end="", flush=True)

    anyio.run(main)
