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
from dataclasses import replace
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
import uuid

import anyio
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    UserPromptPart,
)

from agentpool.agents.acp_agent.session_state import ACPSessionState
from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.events import (
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    ToolResultMetadataEvent,
)
from agentpool.agents.events.processors import FileTracker
from agentpool.agents.modes import ModeInfo
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.models.acp_agents import ACPAgentConfig
from agentpool.utils.streams import merge_queue_into_iterator
from agentpool.utils.subprocess_utils import SubprocessError, monitor_process
from agentpool.utils.token_breakdown import calculate_usage_from_parts


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
    from types import TracebackType

    from anyenv import MultiEventHandler
    from anyio.abc import Process
    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import UserContent
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
    from agentpool.messaging import MessageHistory
    from agentpool.models.acp_agents import BaseACPAgentConfig
    from agentpool.ui.base import InputProvider
    from agentpool_config.nodes import ToolConfirmationMode

logger = get_logger(__name__)


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
    ) -> None:
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge

        # Build config from kwargs if not provided
        if config is None:
            if command is None:
                msg = "Either config or command must be provided"
                raise ValueError(msg)
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
        self._agent_capabilities: AgentCapabilities | None = None
        self._session_id: str | None = None
        self._state: ACPSessionState | None = None
        self.deps_type = type(None)
        self._extra_mcp_servers: list[McpServer] = []
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
        )

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets from config and start bridge if needed."""
        if not self.config.tools:
            return
        # Create providers from tool configs and add to tool manager
        for provider in self.config.get_tool_providers():
            self.tools.add_provider(provider)
        # Start bridge to expose tools via MCP
        await self._tool_bridge.start()
        # Add bridge's MCP server to session
        mcp_config = self._tool_bridge.get_mcp_server_config()
        self._extra_mcp_servers.append(mcp_config)

    async def __aenter__(self) -> Self:
        """Start subprocess and initialize ACP connection."""
        await super().__aenter__()
        await self._setup_toolsets()  # Setup toolsets before session creation
        process = await self._start_process()
        try:
            async with monitor_process(process, context="ACP initialization"):
                await self._initialize()
                await self._create_session()
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
        prompt_manager = self.agent_pool.manifest.prompt_manager if self.agent_pool else None
        args = await self.config.get_args(prompt_manager)
        cmd = [self.config.get_command(), *args]
        self.log.info("Starting ACP subprocess", command=cmd)
        env = {**os.environ, **self.config.env}
        cwd = str(self.config.cwd) if self.config.cwd else None
        self._process = await anyio.open_process(cmd, env=env, cwd=cwd)
        if not self._process.stdin or not self._process.stdout:
            msg = "Failed to create subprocess pipes"
            raise RuntimeError(msg)
        return self._process

    async def _initialize(self) -> None:
        """Initialize the ACP connection."""
        from acp.client.connection import ClientSideConnection
        from agentpool.agents.acp_agent.client_handler import ACPClientHandler

        if not self._process or not self._process.stdin or not self._process.stdout:
            msg = "Process not started"
            raise RuntimeError(msg)

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
        self._agent_capabilities = init_response.agent_capabilities
        # Log MCP capabilities if present
        mcp_caps = self._agent_capabilities.mcp_capabilities if self._agent_capabilities else None
        self.log.info(
            "ACP connection initialized",
            agent_info=self._agent_info,
            mcp_http=mcp_caps.http if mcp_caps else False,
            mcp_sse=mcp_caps.sse if mcp_caps else False,
        )

    async def _create_session(self) -> None:
        """Create a new ACP session with configured MCP servers."""
        from acp.schema import NewSessionRequest
        from acp.schema.mcp import HttpMcpServer, SseMcpServer
        from agentpool.agents.acp_agent.acp_converters import mcp_configs_to_acp

        if not self._connection:
            msg = "Connection not initialized"
            raise RuntimeError(msg)

        # Check which MCP transports the agent supports
        supports_http = (
            self._agent_capabilities
            and self._agent_capabilities.mcp_capabilities
            and self._agent_capabilities.mcp_capabilities.http
        )
        supports_sse = (
            self._agent_capabilities
            and self._agent_capabilities.mcp_capabilities
            and self._agent_capabilities.mcp_capabilities.sse
        )

        # Collect all MCP servers (config + extra) and filter by agent capabilities
        all_servers = self._extra_mcp_servers[:]
        # Add servers from config (converted to ACP format)
        config_servers = self.config.get_mcp_servers()
        if config_servers:
            all_servers.extend(mcp_configs_to_acp(config_servers))
        # Add extra MCP servers (e.g., from tool bridges)
        # Filter servers based on agent transport capabilities
        supported_servers: list[McpServer] = []
        unsupported_servers: list[tuple[McpServer, str]] = []

        for server in all_servers:
            if isinstance(server, HttpMcpServer) and not supports_http:
                unsupported_servers.append((server, "HTTP"))
            elif isinstance(server, SseMcpServer) and not supports_sse:
                unsupported_servers.append((server, "SSE"))
            else:
                # Stdio servers or supported transport types
                supported_servers.append(server)

        if unsupported_servers:
            transports = ", ".join(sorted({t for _, t in unsupported_servers}))
            server_names = ", ".join(s.name for s, _ in unsupported_servers)
            self.log.warning(
                "Agent does not support some MCP transports, skipping servers",
                unsupported_transports=transports,
                skipped_servers=server_names,
                unsupported_count=len(unsupported_servers),
                supported_http=supports_http,
                supported_sse=supports_sse,
            )

        mcp_servers = supported_servers
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
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

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
            """Poll events from ACP state until prompt completes."""
            last_idx = 0
            assert self._state
            while not prompt_task.done():
                if self._client_handler:
                    try:
                        await asyncio.wait_for(
                            self._client_handler._update_event.wait(), timeout=0.05
                        )
                        self._client_handler._update_event.clear()
                    except TimeoutError:
                        pass

                # Yield new events from state
                while last_idx < len(self._state.events):
                    yield self._state.events[last_idx]
                    last_idx += 1

            # Yield remaining events after prompt completes
            while last_idx < len(self._state.events):
                yield self._state.events[last_idx]
                last_idx += 1

        # Accumulate metadata events by tool_call_id (workaround for MCP stripping _meta)
        tool_metadata: dict[str, dict[str, Any]] = {}
        # Merge ACP events with custom events from queue
        # Set deps/input_provider on tool bridge (ContextVar doesn't work - separate task)
        try:
            async with (
                self._tool_bridge.set_run_context(deps, input_provider),
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
                    match event:
                        case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                            text_chunks.append(delta)
                            current_response_parts.append(TextPart(content=delta))
                        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta)) if delta:
                            current_response_parts.append(ThinkingPart(content=delta))
                        case ToolCallStartEvent(
                            tool_call_id=tc_id, tool_name=tc_name, raw_input=tc_input
                        ):
                            current_response_parts.append(
                                ToolCallPart(tool_name=tc_name, args=tc_input, tool_call_id=tc_id)
                            )

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
        """Update the model for the current session via ACP protocol.

        Attempts to use the ACP protocol to change the model:
        1. If config_options exist with a 'model' category, use set_session_config_option
        2. Otherwise, use legacy set_session_model API

        Args:
            model: New model ID to use

        Raises:
            RuntimeError: If no active session or remote agent doesn't support model changes
        """
        from acp.schema import SetSessionConfigOptionRequest, SetSessionModelRequest

        if not self._connection or not self._session_id:
            msg = "Cannot set model: no active session"
            raise RuntimeError(msg)

        if not self._state:
            msg = "Cannot set model: no session state"
            raise RuntimeError(msg)

        # Try using the new unified config options API first
        model_cfg = next((i for i in self._state.config_options if i.category == "model"), None)
        if model_cfg:
            # Use new unified API
            request = SetSessionConfigOptionRequest(
                session_id=self._session_id,
                config_id=model_cfg.id,
                value=model,
            )
            response = await self._connection.set_session_config_option(request)
            if response:
                # Update entire config_options state from response
                self._state.config_options = list(response.config_options)
                self.log.info("Model changed via SessionConfigOption", model=model)
                return
            msg = "set_session_config_option returned no response"
            raise RuntimeError(msg)

        # Fallback to legacy set_session_model API
        request_legacy = SetSessionModelRequest(session_id=self._session_id, model_id=model)
        response_legacy = await self._connection.set_session_model(request_legacy)
        if response_legacy:
            # Update legacy state
            self._state.current_model_id = model
            self.log.info("Model changed via legacy set_session_model", model=model)
            return

        # If we get here, the remote agent doesn't support model changes
        msg = (
            "Remote ACP agent does not support model changes. "
            "No config_options with category='model' found and set_session_model "
            "returned no response."
        )
        raise RuntimeError(msg)

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
        # Update client handler if it exists
        if self._client_handler:
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

    async def interrupt(self) -> None:
        """Interrupt the currently running stream.

        Sends a CancelNotification to the remote ACP server and cancels
        the local prompt task.
        """
        from acp.schema import CancelNotification

        self._cancelled = True
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

    async def set_mode(self, mode: ModeInfo | str, category_id: str | None = None) -> None:
        """Set a mode on the remote ACP server.

        For ACPAgent, this forwards the mode/model change to the remote ACP server.
        Prefers new set_session_config_option if config_options are available,
        falls back to legacy set_session_mode/set_session_model.

        Args:
            mode: The mode to set - ModeInfo object or mode ID string
            category_id: Category ID (config option ID)

        Raises:
            RuntimeError: If not connected to ACP server
            ValueError: If mode is not available
        """
        from acp.schema import (
            SetSessionConfigOptionRequest,
            SetSessionModelRequest,
            SetSessionModeRequest,
        )

        # Extract mode_id and category from ModeInfo if provided
        if isinstance(mode, ModeInfo):
            mode_id = mode.id
            category_id = category_id or mode.category_id
        else:
            mode_id = mode

        if not self._connection or not self._session_id or not self._state:
            msg = "Not connected to ACP server"
            raise RuntimeError(msg)

        # Validate mode is available
        available_modes = await self.get_modes()
        matching_category = (
            next((c for c in available_modes if c.id == category_id), None) if category_id else None
        )

        if matching_category:
            valid_ids = {m.id for m in matching_category.available_modes}
            if mode_id not in valid_ids:
                msg = f"Unknown {category_id}: {mode_id}. Available: {valid_ids}"
                raise ValueError(msg)
        elif category_id:
            # Category specified but not found
            available_cats = {c.id for c in available_modes}
            msg = f"Unknown category: {category_id}. Available: {available_cats}"
            raise ValueError(msg)
        else:
            # No category specified and no match found
            msg = "category_id is required when mode is a string"
            raise ValueError(msg)

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
            self.log.info("ACP server Config option changed", config_id=category_id, value=mode_id)
            return

        # Legacy: Use old set_session_mode/set_session_model APIs
        if category_id == "permissions":
            mode_request = SetSessionModeRequest(session_id=self._session_id, mode_id=mode_id)
            await self._connection.set_session_mode(mode_request)
            # Update local state
            if self._state.modes:
                self._state.modes.current_mode_id = mode_id
            self.log.info("Mode changed on remote ACP server (legacy)", mode_id=mode_id)

        elif category_id == "model":
            model_request = SetSessionModelRequest(session_id=self._session_id, model_id=mode_id)
            await self._connection.set_session_model(model_request)
            # Update local state
            if self._state.models:
                self._state.models.current_model_id = mode_id
            self.log.info("Model changed on remote ACP server (legacy)", model_id=mode_id)

        else:
            msg = f"Unknown category: {category_id}. Available: permissions, model"
            raise ValueError(msg)


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
