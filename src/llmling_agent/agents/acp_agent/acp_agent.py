"""ACP Agent - MessageNode wrapping an external ACP subprocess.

This module provides an agent implementation that communicates with external
ACP (Agent Client Protocol) servers via stdio, enabling integration of any
ACP-compatible agent into the llmling-agent pool.

The ACPAgent class acts as an ACP client, spawning an ACP server subprocess
and communicating with it via JSON-RPC over stdio. This allows:
- Integration of external ACP-compatible agents (like claude-code-acp)
- Composition with native llmling agents via connections, teams, etc.
- Full ACP protocol support including file operations and terminals

Example:
    ```python
    config = ACPAgentConfig(
        command="claude-code-acp",
        name="claude_coder",
        cwd="/path/to/project",
    )
    async with ACPAgent(config) as agent:
        result = await agent.run("Write a hello world program")
        print(result.content)
    ```
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload
import uuid

from anyenv import MultiEventHandler, create_process
from pydantic_ai import PartDeltaEvent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    UserPromptPart,
)

from acp.client.connection import ClientSideConnection
from acp.schema import InitializeRequest, NewSessionRequest, PromptRequest
from acp.utils import to_acp_content_blocks
from llmling_agent.agents.acp_agent.acp_converters import convert_to_acp_content, mcp_configs_to_acp
from llmling_agent.agents.acp_agent.client_handler import ACPClientHandler
from llmling_agent.agents.acp_agent.session_state import ACPSessionState
from llmling_agent.agents.base_agent import BaseAgent
from llmling_agent.agents.events import RunStartedEvent, StreamCompleteEvent, ToolCallStartEvent
from llmling_agent.common_types import IndividualEventHandler
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage, MessageHistory
from llmling_agent.messaging.processing import prepare_prompts
from llmling_agent.models.acp_agents import ACPAgentConfig, MCPCapableACPAgentConfig
from llmling_agent.talk.stats import MessageStats


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
    from types import TracebackType

    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import FinishReason
    from tokonomics.model_discovery import ProviderType

    from acp.agent.protocol import Agent as ACPAgentProtocol
    from acp.client.protocol import Client
    from acp.schema import (
        InitializeResponse,
        RequestPermissionRequest,
        RequestPermissionResponse,
        StopReason,
    )
    from acp.schema.mcp import McpServer
    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import BuiltinEventHandlerType, PromptCompatible, SimpleJsonType
    from llmling_agent.delegation import AgentPool
    from llmling_agent.mcp_server.tool_bridge import ToolManagerBridge
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.acp_agents import BaseACPAgentConfig
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import ToolConfirmationMode

logger = get_logger(__name__)

PROTOCOL_VERSION = 1

STOP_REASON_MAP: dict[StopReason, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "content_filter",
    "cancelled": "error",
}


def extract_file_path_from_tool_call(tool_name: str, raw_input: dict[str, Any]) -> str | None:
    """Extract file path from a tool call if it's a file-writing tool.

    Uses simple heuristics by default:
    - Tool name contains 'write' or 'edit' (case-insensitive)
    - Input contains 'path' or 'file_path' key

    Override in subclasses for agent-specific tool naming conventions.

    Args:
        tool_name: Name of the tool being called
        raw_input: Tool call arguments

    Returns:
        File path if this is a file-writing tool, None otherwise
    """
    name_lower = tool_name.lower()
    if "write" not in name_lower and "edit" not in name_lower:
        return None

    # Try common path argument names
    for key in ("file_path", "path", "filepath", "filename", "file"):
        if key in raw_input and isinstance(val := raw_input[key], str):
            return val

    return None


class ACPAgent[TDeps = None](BaseAgent[TDeps, str]):
    """MessageNode that wraps an external ACP agent subprocess.

    This allows integrating any ACP-compatible agent into the llmling-agent
    pool, enabling composition with native agents via connections, teams, etc.

    The agent manages:
    - Subprocess lifecycle (spawn on enter, terminate on exit)
    - ACP protocol initialization and session creation
    - Prompt execution with session update collection
    - Client-side operations (filesystem, terminals, permissions)

    Supports both blocking `run()` and streaming `run_iter()` execution modes.

    Example with config:
        ```python
        config = ClaudeACPAgentConfig(cwd="/project", model="sonnet")
        agent = ACPAgent(config, agent_pool=pool)
        ```

    Example with kwargs:
        ```python
        agent = ACPAgent(
            command="claude-code-acp",
            cwd="/project",
            providers=["anthropic"],
        )
        ```
    """

    @overload
    def __init__(
        self,
        *,
        config: BaseACPAgentConfig,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        command: str,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        env: ExecutionEnvironment | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        providers: list[ProviderType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "always",
    ) -> None: ...

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
        env: ExecutionEnvironment | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        providers: list[ProviderType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "always",
    ) -> None:
        from llmling_agent.agents.events import resolve_event_handlers
        from llmling_agent.tools.manager import ToolManager

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
                providers=list(providers) if providers else [],
            )
        super().__init__(
            name=name or config.name or config.get_command(),
            description=description,
            display_name=display_name,
            mcp_servers=config.mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs or list(config.triggers),
        )
        self.acp_permission_callback: (
            Callable[[RequestPermissionRequest], Awaitable[RequestPermissionResponse]] | None
        ) = None
        self.config = config
        self.env = env or config.get_execution_environment()
        self._input_provider = input_provider
        self._process: Process | None = None
        self._connection: ClientSideConnection | None = None
        self._client_handler: ACPClientHandler | None = None
        self._init_response: InitializeResponse | None = None
        self._session_id: str | None = None
        self._state: ACPSessionState | None = None
        self._output_type = str
        self.conversation = MessageHistory()
        self.deps_type = type(None)
        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = asyncio.Queue()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler = MultiEventHandler[IndividualEventHandler](resolved_handlers)
        self._extra_mcp_servers: list[McpServer] = []
        # Initialize ToolManager for toolsets (read-only, for bridge exposure)
        self.tools = ToolManager()
        self._tool_bridge: ToolManagerBridge | None = None
        self._owns_bridge = False  # Track if we created the bridge (for cleanup)

        # Copy tool confirmation mode from config (aligned with Agent class)
        # auto_grant_permissions=True maps to "never", False maps to "always"
        self.tool_confirmation_mode: ToolConfirmationMode = tool_confirmation_mode

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        from llmling_agent.messaging.context import NodeContext
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        cfg = NodeConfig(name=self.name, description=self.description)
        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return NodeContext(node=self, pool=self.agent_pool, config=cfg, definition=defn)

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets from config and create bridge if needed."""
        from llmling_agent.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge

        if not isinstance(self.config, MCPCapableACPAgentConfig) or not self.config.toolsets:
            return
        # Create providers from toolset configs and add to tool manager
        for toolset_config in self.config.toolsets:
            provider = toolset_config.get_provider()
            self.tools.add_provider(provider)
        # Auto-create bridge to expose tools via MCP
        config = BridgeConfig(transport="sse", server_name=f"llmling-{self.name}-tools")
        self._tool_bridge = ToolManagerBridge(node=self, config=config)
        await self._tool_bridge.start()
        self._owns_bridge = True
        # Add bridge's MCP server to session
        mcp_config = self._tool_bridge.get_mcp_server_config()
        self._extra_mcp_servers.append(mcp_config)

    async def __aenter__(self) -> Self:
        """Start subprocess and initialize ACP connection."""
        await super().__aenter__()
        await self._setup_toolsets()  # Setup toolsets before session creation
        await self._start_process()
        await self._initialize()
        await self._create_session()
        await asyncio.sleep(0.3)  # Small delay to let subprocess fully initialize
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

    async def _start_process(self) -> None:
        """Start the ACP server subprocess."""
        cmd = [self.config.get_command(), *self.config.get_args()]
        self.log.info("Starting ACP subprocess", command=cmd)
        self._process = await create_process(
            *cmd,
            stdin="pipe",
            stdout="pipe",
            stderr="pipe",
            env={**os.environ, **self.config.env},
            cwd=self.config.cwd,
        )
        if not self._process.stdin or not self._process.stdout:
            msg = "Failed to create subprocess pipes"
            raise RuntimeError(msg)

    async def _initialize(self) -> None:
        """Initialize the ACP connection."""
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
        init_request = InitializeRequest.create(
            title="LLMling Agent",
            version="0.1.0",
            name="llmling-agent",
            protocol_version=PROTOCOL_VERSION,
            terminal=self.config.allow_terminal,
            read_text_file=self.config.allow_file_operations,
            write_text_file=self.config.allow_file_operations,
        )
        self._init_response = await self._connection.initialize(init_request)
        self.log.info("ACP connection initialized", agent_info=self._init_response.agent_info)

    async def _create_session(self) -> None:
        """Create a new ACP session with configured MCP servers."""
        if not self._connection:
            msg = "Connection not initialized"
            raise RuntimeError(msg)

        mcp_servers: list[McpServer] = []  # Collect MCP servers from config
        # Add servers from config (converted to ACP format)
        config_servers = self.config.get_mcp_servers()
        if config_servers:
            mcp_servers.extend(mcp_configs_to_acp(config_servers))
        # Add any extra MCP servers (e.g., from tool bridges)
        mcp_servers.extend(self._extra_mcp_servers)
        cwd = self.config.cwd or str(Path.cwd())
        session_request = NewSessionRequest(cwd=cwd, mcp_servers=mcp_servers)
        response = await self._connection.new_session(session_request)
        self._session_id = response.session_id
        if self._state:
            self._state.session_id = self._session_id
            if response.models:  # Store full model info from session response
                self._state.models = response.models
                self._state.current_model_id = response.models.current_model_id
            self._state.modes = response.modes
        model = self._state.current_model_id if self._state else None
        self.log.info("ACP session created", session_id=self._session_id, model=model)

    def add_mcp_server(self, server: McpServer) -> None:
        """Add an MCP server to be passed to the next session."""
        self._extra_mcp_servers.append(server)

    async def add_tool_bridge(self, bridge: ToolManagerBridge) -> None:
        """Add an external tool bridge to expose its tools via MCP.

        The bridge must already be started. Its MCP server config will be
        added to the session. Use this for bridges created externally
        (e.g., from AgentPool). For toolsets defined in config, bridges
        are created automatically.

        Args:
            bridge: Started ToolManagerBridge instance
        """
        if self._tool_bridge is None:  # Don't replace our own bridge
            self._tool_bridge = bridge
        mcp_config = bridge.get_mcp_server_config()
        self._extra_mcp_servers.append(mcp_config)
        self.log.info("Added external tool bridge", url=bridge.url)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._tool_bridge and self._owns_bridge:  # Stop our own bridge if we created it
            await self._tool_bridge.stop()
        self._tool_bridge = None
        self._owns_bridge = False
        self._extra_mcp_servers.clear()

        if self._client_handler:
            try:
                await self._client_handler.cleanup()
            except Exception:
                self.log.exception("Error cleaning up client handler")
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

    async def run(
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
    ) -> ChatMessage[str]:
        """Execute prompt against ACP agent.

        Args:
            prompts: Prompts to send (will be joined with spaces)
            message_id: Optional message id for the returned message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own

        Returns:
            ChatMessage containing the agent's aggregated text response
        """
        # Collect all events through run_stream
        final_message: ChatMessage[str] | None = None
        async for event in self.run_stream(
            *prompts,
            message_id=message_id,
            input_provider=input_provider,
            message_history=message_history,
        ):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        if final_message is None:
            msg = "No final message received from stream"
            raise RuntimeError(msg)

        return final_message

    async def run_stream(  # noqa: PLR0915
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        """Stream native events as they arrive from ACP agent.

        Args:
            prompts: Prompts to send (will be joined with spaces)
            message_id: Optional message id for the final message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own

        Yields:
            RichAgentStreamEvent instances converted from ACP session updates
        """
        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider
            if self._client_handler:
                self._client_handler._input_provider = input_provider
        if not self._connection or not self._session_id or not self._state:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        conversation = message_history if message_history is not None else self.conversation
        # Prepare user message for history and convert to ACP content blocks
        user_msg, processed_prompts, _original_message = await prepare_prompts(*prompts)
        run_id = str(uuid.uuid4())
        self._state.clear()  # Reset state

        # Track messages in pydantic-ai format: ModelRequest -> ModelResponse -> ...
        # This mirrors pydantic-ai's new_messages() which includes the initial user request.

        model_messages: list[ModelResponse | ModelRequest] = []
        # Start with the user's request (same as pydantic-ai's new_messages())
        initial_request = ModelRequest(parts=[UserPromptPart(content=processed_prompts)])
        model_messages.append(initial_request)
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []  # For final content string
        touched_files: set[str] = set()  # Track files modified by tool calls
        run_started = RunStartedEvent(
            thread_id=self.conversation_id,
            run_id=run_id,
            agent_name=self.name,
        )
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, run_started)
        yield run_started
        content_blocks = convert_to_acp_content(processed_prompts)
        pending_parts = conversation.get_pending_parts()
        final_blocks = [*to_acp_content_blocks(pending_parts), *content_blocks]
        prompt_request = PromptRequest(session_id=self._session_id, prompt=final_blocks)
        self.log.debug("Starting streaming prompt", num_blocks=len(final_blocks))
        # Run prompt in background
        prompt_task = asyncio.create_task(self._connection.prompt(prompt_request))
        last_idx = 0
        while not prompt_task.done():
            if self._client_handler:
                try:  # Wait for new events
                    await asyncio.wait_for(self._client_handler._update_event.wait(), timeout=0.05)
                    self._client_handler._update_event.clear()
                except TimeoutError:
                    pass

            # Yield new native events and distribute to handlers
            while last_idx < len(self._state.events):
                event = self._state.events[last_idx]
                # Check for queued custom events first
                while not self._event_queue.empty():
                    try:
                        custom_event = self._event_queue.get_nowait()
                        for handler in self.event_handler._wrapped_handlers:
                            await handler(None, custom_event)
                        yield custom_event
                    except asyncio.QueueEmpty:
                        break

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
                        # Track files modified by write/edit tools
                        if file_path := extract_file_path_from_tool_call(
                            tc_name or "", tc_input or {}
                        ):
                            touched_files.add(file_path)

                for handler in self.event_handler._wrapped_handlers:  # Distribute to handlers
                    await handler(None, event)
                yield event
                last_idx += 1

        # Yield remaining events after completion
        while last_idx < len(self._state.events):
            event = self._state.events[last_idx]
            # Extract content from events and build parts in arrival order
            match event:
                case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                    text_chunks.append(delta)
                    current_response_parts.append(TextPart(content=delta))
                case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta)) if delta:
                    current_response_parts.append(ThinkingPart(content=delta))
                case ToolCallStartEvent(tool_call_id=tc_id, tool_name=tc_name, raw_input=tc_input):
                    current_response_parts.append(
                        ToolCallPart(tool_name=tc_name, args=tc_input, tool_call_id=tc_id)
                    )
                    # Track files modified by write/edit tools
                    if file_path := extract_file_path_from_tool_call(tc_name or "", tc_input or {}):
                        touched_files.add(file_path)

            for handler in self.event_handler._wrapped_handlers:
                await handler(None, event)
            yield event
            last_idx += 1
        # Ensure we catch any exceptions from the prompt task
        response = await prompt_task
        finish_reason: FinishReason = STOP_REASON_MAP.get(response.stop_reason, "stop")
        # Flush response parts to model_messages
        if current_response_parts:
            model_messages.append(ModelResponse(parts=current_response_parts))

        text_content = "".join(text_chunks)
        # Build metadata with touched files if any
        metadata: SimpleJsonType = {}
        if touched_files:
            metadata["touched_files"] = sorted(touched_files)
        message = ChatMessage[str](
            content=text_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            model_name=self.model_name,
            messages=model_messages,
            metadata=metadata,
            finish_reason=finish_reason,
        )
        complete_event = StreamCompleteEvent(message=message)
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, complete_event)
        yield complete_event  # Emit final StreamCompleteEvent with aggregated message
        self.message_sent.emit(message)
        conversation.add_chat_messages([user_msg, message])  # Record to conversation history

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
    ) -> AsyncIterator[ChatMessage[str]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially

        Yields:
            Response messages in sequence
        """
        for prompts in prompt_groups:
            response = await self.run(*prompts)
            yield response

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        if self._state and self._state.current_model_id:
            return self._state.current_model_id
        if self._init_response and self._init_response.agent_info:
            return self._init_response.agent_info.name
        return None

    async def set_model(self, model: str) -> None:
        """Update the model and restart the ACP agent process.

        Args:
            model: New model name to use

        Raises:
            ValueError: If the config doesn't have a model field
            RuntimeError: If agent is currently processing (has active process but no session)
        """
        # TODO: Once ACP protocol stabilizes, use set_session_model instead of restart
        # from acp.schema import SetSessionModelRequest  # UNSTABLE
        # if self._connection and self._session_id:
        #     request = SetSessionModelRequest(session_id=self._session_id, model_id=model)
        #     await self._connection.set_session_model(request)
        #     if self._state:
        #         self._state.current_model_id = model
        #     self.log.info("Model changed via ACP protocol", model=model)
        #     return

        if not hasattr(self.config, "model"):
            msg = f"Config type {type(self.config).__name__} doesn't support model changes"
            raise ValueError(msg)
        # Prevent changes during active processing
        if self._process and not self._session_id:
            msg = "Cannot change model while agent is initializing"
            raise RuntimeError(msg)
        # Create new config with updated model
        new_config = self.config.model_copy(update={"model": model})
        if self._process:  # Clean up existing process if any
            await self._cleanup()
        self.config = new_config  # Update config and restart
        await self._start_process()
        await self._initialize()

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set the tool confirmation mode for this agent.

        For ACPAgent, this sends a set_session_mode request to the remote ACP server
        to change its mode. The mode is also stored locally for the client handler.

        Note: "per_tool" behaves like "always" since we don't have per-tool metadata
        from the ACP server.

        Args:
            mode: Tool confirmation mode
        """
        from llmling_agent_server.acp_server.converters import confirmation_mode_to_mode_id

        self.tool_confirmation_mode = mode
        # Update client handler if it exists
        if self._client_handler:
            self._client_handler.tool_confirmation_mode = mode

        # Forward mode change to remote ACP server if connected
        if self._connection and self._session_id:
            from acp.schema import SetSessionModeRequest

            mode_id = confirmation_mode_to_mode_id(mode)
            request = SetSessionModeRequest(session_id=self._session_id, mode_id=mode_id)
            try:
                await self._connection.set_session_mode(request)
                self.log.info(
                    "Forwarded mode change to remote ACP server",
                    mode=mode,
                    mode_id=mode_id,
                )
            except Exception:
                self.log.exception("Failed to forward mode change to remote ACP server")
        else:
            self.log.info("Tool confirmation mode changed (local only)", mode=mode)

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        return MessageStats(messages=list(self.conversation.chat_messages))


if __name__ == "__main__":

    async def main() -> None:
        """Demo: Basic call to an ACP agent."""
        args = ["run", "llmling-agent", "serve-acp", "--model-provider", "openai"]
        cwd = str(Path.cwd())
        async with ACPAgent(command="uv", args=args, cwd=cwd, event_handlers=["detailed"]) as agent:
            print("Response (streaming): ", end="", flush=True)
            async for chunk in agent.run_stream("Say hello briefly."):
                print(chunk, end="", flush=True)

    asyncio.run(main())
