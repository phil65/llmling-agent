"""ClaudeCodeAgent - Native Claude Agent SDK integration.

This module provides an agent implementation that wraps the Claude Agent SDK's
ClaudeSDKClient for native integration with agentpool.

The ClaudeCodeAgent acts as a client to the Claude Code CLI, enabling:
- Bidirectional streaming communication
- Tool permission handling via callbacks
- Integration with agentpool's event system

Example:
    ```python
    async with ClaudeCodeAgent(
        name="claude_coder",
        cwd="/path/to/project",
        allowed_tools=["Read", "Write", "Bash"],
    ) as agent:
        async for event in agent.run_stream("Write a hello world program"):
            print(event)
    ```
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Self, cast
import uuid

from pydantic import TypeAdapter
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunUsage,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.claude_code_agent.converters import claude_message_to_events
from agentpool.agents.events import (
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.messaging.messages import TokenCost
from agentpool.messaging.processing import prepare_prompts
from agentpool.models.claude_code_agents import ClaudeCodeAgentConfig
from agentpool.utils.streams import merge_queue_into_iterator


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        McpServerConfig,
        PermissionMode,
        PermissionResult,
        ToolPermissionContext,
        ToolUseBlock,
    )
    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment
    from toprompt import AnyPromptType

    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        IndividualEventHandler,
        PromptCompatible,
    )
    from agentpool.delegation import AgentPool
    from agentpool.mcp_server.tool_bridge import ToolManagerBridge
    from agentpool.messaging import MessageHistory
    from agentpool.talk.stats import MessageStats
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode


logger = get_logger(__name__)


class ClaudeCodeAgent[TDeps = None, TResult = str](BaseAgent[TDeps, TResult]):
    """Agent wrapping Claude Agent SDK's ClaudeSDKClient.

    This provides native integration with Claude Code, enabling:
    - Bidirectional streaming for interactive conversations
    - Tool permission handling via can_use_tool callback
    - Full access to Claude Code's capabilities (file ops, terminals, etc.)

    The agent manages:
    - ClaudeSDKClient lifecycle (connect on enter, disconnect on exit)
    - Event conversion from Claude SDK to agentpool events
    - Tool confirmation via input provider
    """

    def __init__(
        self,
        *,
        config: ClaudeCodeAgentConfig | None = None,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        cwd: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        system_prompt: str | Sequence[str] | None = None,
        include_builtin_system_prompt: bool = True,
        model: str | None = None,
        max_turns: int | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: PermissionMode | None = None,
        mcp_servers: Sequence[MCPServerConfig] | None = None,
        environment: dict[str, str] | None = None,
        add_dir: list[str] | None = None,
        builtin_tools: list[str] | None = None,
        fallback_model: str | None = None,
        dangerously_skip_permissions: bool = False,
        env: ExecutionEnvironment | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "always",
        output_type: type[TResult] | None = None,
    ) -> None:
        """Initialize ClaudeCodeAgent.

        Args:
            config: Configuration object (alternative to individual kwargs)
            name: Agent name
            description: Agent description
            display_name: Display name for UI
            cwd: Working directory for Claude Code
            allowed_tools: List of allowed tool names
            disallowed_tools: List of disallowed tool names
            system_prompt: System prompt - string or list (appended to builtin by default)
            include_builtin_system_prompt: If True, the builtin system prompt is included.
            model: Model to use (e.g., "claude-sonnet-4-5")
            max_turns: Maximum conversation turns
            max_thinking_tokens: Max tokens for extended thinking
            permission_mode: Permission mode ("default", "acceptEdits", "plan", "bypassPermissions")
            mcp_servers: External MCP servers to connect to (internal format, converted at runtime)
            environment: Environment variables for the agent process
            add_dir: Additional directories to allow tool access to
            builtin_tools: Available tools from Claude Code's built-in set (empty list disables all)
            fallback_model: Fallback model when default is overloaded
            dangerously_skip_permissions: Bypass all permission checks (sandboxed only)
            env: Execution environment
            input_provider: Provider for user input/confirmations
            agent_pool: Agent pool for multi-agent coordination
            enable_logging: Whether to enable logging
            event_configs: Event configuration
            event_handlers: Event handlers for streaming events
            tool_confirmation_mode: Tool confirmation behavior
            output_type: Type for structured output (uses JSON schema)
        """
        from agentpool.agents.sys_prompts import SystemPrompts

        # Build config from kwargs if not provided
        if config is None:
            config = ClaudeCodeAgentConfig(
                name=name or "claude_code",
                description=description,
                display_name=display_name,
                cwd=cwd,
                model=model,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                system_prompt=system_prompt,
                include_builtin_system_prompt=include_builtin_system_prompt,
                max_turns=max_turns,
                max_thinking_tokens=max_thinking_tokens,
                permission_mode=permission_mode,
                mcp_servers=list(mcp_servers) if mcp_servers else [],
                env=environment,
                add_dir=add_dir,
                builtin_tools=builtin_tools,
                fallback_model=fallback_model,
                dangerously_skip_permissions=dangerously_skip_permissions,
            )

        super().__init__(
            name=name or config.name or "claude_code",
            description=description or config.description,
            display_name=display_name or config.display_name,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs or list(config.triggers),
            env=env,
            input_provider=input_provider,
            output_type=output_type or str,  # type: ignore[arg-type]
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
        )

        self._config = config
        self._cwd = cwd or config.cwd
        self._allowed_tools = allowed_tools or config.allowed_tools
        self._disallowed_tools = disallowed_tools or config.disallowed_tools
        self._include_builtin_system_prompt = (
            include_builtin_system_prompt and config.include_builtin_system_prompt
        )

        # Initialize SystemPrompts manager
        # Normalize system_prompt to a list
        all_prompts: list[AnyPromptType] = []
        prompt_source = system_prompt if system_prompt is not None else config.system_prompt
        if prompt_source is not None:
            if isinstance(prompt_source, str):
                all_prompts.append(prompt_source)
            else:
                all_prompts.extend(prompt_source)
        prompt_manager = agent_pool.manifest.prompt_manager if agent_pool else None
        self.sys_prompts = SystemPrompts(all_prompts, prompt_manager=prompt_manager)
        self._model = model or config.model
        self._max_turns = max_turns or config.max_turns
        self._max_thinking_tokens = max_thinking_tokens or config.max_thinking_tokens
        self._permission_mode: PermissionMode | None = permission_mode or config.permission_mode
        self._external_mcp_servers = list(mcp_servers) if mcp_servers else config.get_mcp_servers()
        self._environment = environment or config.env
        self._add_dir = add_dir or config.add_dir
        self._builtin_tools = builtin_tools if builtin_tools is not None else config.builtin_tools
        self._fallback_model = fallback_model or config.fallback_model
        self._dangerously_skip_permissions = (
            dangerously_skip_permissions or config.dangerously_skip_permissions
        )

        # Client state
        self._client: ClaudeSDKClient | None = None
        self._current_model: str | None = self._model
        self.deps_type = type(None)

        # ToolBridge state for exposing toolsets via MCP
        self._tool_bridge: ToolManagerBridge | None = None
        self._owns_bridge = False  # Track if we created the bridge (for cleanup)
        self._mcp_servers: dict[str, McpServerConfig] = {}  # Claude SDK MCP server configs

    def get_context(self, data: Any = None) -> AgentContext:
        """Create a new context for this agent.

        Args:
            data: Optional custom data to attach to the context

        Returns:
            A new AgentContext instance
        """
        from agentpool.agents import AgentContext
        from agentpool.models import AgentsManifest

        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return AgentContext(
            node=self, pool=self.agent_pool, config=self._config, definition=defn, data=data
        )

    def _convert_mcp_servers_to_sdk_format(self) -> dict[str, McpServerConfig]:
        """Convert internal MCPServerConfig to Claude SDK format.

        Returns:
            Dict mapping server names to SDK-compatible config dicts
        """
        from claude_agent_sdk import McpServerConfig

        from agentpool_config.mcp_server import (
            SSEMCPServerConfig,
            StdioMCPServerConfig,
            StreamableHTTPMCPServerConfig,
        )

        result: dict[str, McpServerConfig] = {}

        for idx, server in enumerate(self._external_mcp_servers):
            # Determine server name
            if server.name:
                name = server.name
            elif isinstance(server, StdioMCPServerConfig) and server.args:
                name = server.args[-1].split("/")[-1].split("@")[0]
            elif isinstance(server, StdioMCPServerConfig):
                name = server.command
            elif isinstance(server, SSEMCPServerConfig | StreamableHTTPMCPServerConfig):
                from urllib.parse import urlparse

                name = urlparse(str(server.url)).hostname or f"server_{idx}"
            else:
                name = f"server_{idx}"

            # Build SDK-compatible config
            config: dict[str, Any]
            match server:
                case StdioMCPServerConfig(command=command, args=args):
                    config = {"type": "stdio", "command": command, "args": args}
                    if server.env:
                        config["env"] = server.get_env_vars()
                case SSEMCPServerConfig(url=url):
                    config = {"type": "sse", "url": str(url)}
                    if server.headers:
                        config["headers"] = server.headers
                case StreamableHTTPMCPServerConfig(url=url):
                    config = {"type": "http", "url": str(url)}
                    if server.headers:
                        config["headers"] = server.headers

            result[name] = cast(McpServerConfig, config)

        return result

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets from config and create bridge if needed.

        Creates providers from toolset configs, adds them to the tool manager,
        and starts an MCP bridge to expose them to Claude Code via the SDK's
        native MCP support. Also converts external MCP servers to SDK format.
        """
        from agentpool.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge

        # Convert external MCP servers to SDK format first
        if self._external_mcp_servers:
            external_configs = self._convert_mcp_servers_to_sdk_format()
            self._mcp_servers.update(external_configs)
            self.log.info("External MCP servers configured", server_count=len(external_configs))

        if not self._config.toolsets:
            return

        # Create providers from toolset configs and add to tool manager
        for toolset_config in self._config.toolsets:
            provider = toolset_config.get_provider()
            self.tools.add_provider(provider)

        # Auto-create bridge to expose tools via MCP
        config = BridgeConfig(
            transport="streamable-http", server_name=f"agentpool-{self.name}-tools"
        )
        self._tool_bridge = ToolManagerBridge(node=self, config=config)
        await self._tool_bridge.start()
        self._owns_bridge = True

        # Get Claude SDK-compatible MCP config and merge into our servers dict
        mcp_config = self._tool_bridge.get_claude_mcp_server_config()
        self._mcp_servers.update(mcp_config)
        self.log.info("Toolsets initialized", toolset_count=len(self._config.toolsets))

    async def add_tool_bridge(self, bridge: ToolManagerBridge) -> None:
        """Add an external tool bridge to expose its tools via MCP.

        The bridge must already be started. Its MCP server config will be
        added to the Claude SDK options. Use this for bridges created externally
        (e.g., from AgentPool). For toolsets defined in config, bridges
        are created automatically.

        Args:
            bridge: Started ToolManagerBridge instance
        """
        if self._tool_bridge is None:  # Don't replace our own bridge
            self._tool_bridge = bridge

        # Get Claude SDK-compatible config and merge
        mcp_config = bridge.get_claude_mcp_server_config()
        self._mcp_servers.update(mcp_config)
        self.log.info("Added external tool bridge", server_name=bridge.config.server_name)

    async def _cleanup_bridge(self) -> None:
        """Clean up tool bridge resources."""
        if self._tool_bridge and self._owns_bridge:
            await self._tool_bridge.stop()
        self._tool_bridge = None
        self._owns_bridge = False
        self._mcp_servers.clear()

    @property
    def model_name(self) -> str | None:
        """Get the model name."""
        return self._current_model

    def _build_options(self, *, formatted_system_prompt: str | None = None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from runtime state.

        Args:
            formatted_system_prompt: Pre-formatted system prompt from SystemPrompts manager
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk.types import SystemPromptPreset

        # Build system prompt value
        system_prompt: str | SystemPromptPreset | None = None
        if formatted_system_prompt:
            if self._include_builtin_system_prompt:
                # Use SystemPromptPreset to append to builtin prompt
                system_prompt = SystemPromptPreset(
                    type="preset",
                    preset="claude_code",
                    append=formatted_system_prompt,
                )
            else:
                system_prompt = formatted_system_prompt

        # Determine effective permission mode
        permission_mode = self._permission_mode
        if self._dangerously_skip_permissions and not permission_mode:
            permission_mode = "bypassPermissions"

        # Determine can_use_tool callback
        bypass = permission_mode == "bypassPermissions" or self._dangerously_skip_permissions
        can_use_tool = (
            self._can_use_tool if self.tool_confirmation_mode != "never" and not bypass else None
        )

        # Build structured output format if needed
        output_format: dict[str, Any] | None = None
        if self._output_type is not str:
            adapter = TypeAdapter(self._output_type)
            schema = adapter.json_schema()
            output_format = {"type": "json_schema", "schema": schema}

        return ClaudeAgentOptions(
            cwd=self._cwd,
            allowed_tools=self._allowed_tools or [],
            disallowed_tools=self._disallowed_tools or [],
            system_prompt=system_prompt,
            model=self._model,
            max_turns=self._max_turns,
            max_thinking_tokens=self._max_thinking_tokens,
            permission_mode=permission_mode,
            env=self._environment or {},
            add_dirs=self._add_dir or [],  # type: ignore[arg-type]  # SDK uses list not Sequence
            tools=self._builtin_tools,
            fallback_model=self._fallback_model,
            can_use_tool=can_use_tool,
            output_format=output_format,
            mcp_servers=self._mcp_servers or {},
            include_partial_messages=True,
        )

    async def _can_use_tool(  # noqa: PLR0911
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResult:
        """Handle tool permission requests.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input arguments
            context: Permission context with suggestions

        Returns:
            PermissionResult indicating allow or deny
        """
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        from agentpool.tools.base import Tool

        # Auto-grant if confirmation mode is "never"
        if self.tool_confirmation_mode == "never":
            return PermissionResultAllow()

        # Auto-grant tools from our own bridge - they already show ToolCallStartEvent in UI
        # Bridge tools are named like: mcp__agentpool-{agent_name}-tools__{tool}
        if self._tool_bridge:
            bridge_prefix = f"mcp__{self._tool_bridge.config.server_name}__"
            if tool_name.startswith(bridge_prefix):
                return PermissionResultAllow()

        # Use input provider if available
        if self._input_provider:
            # Create a dummy Tool for the confirmation dialog
            desc = f"Claude Code tool: {tool_name}"
            tool = Tool(callable=lambda: None, name=tool_name, description=desc)
            result = await self._input_provider.get_tool_confirmation(
                context=self.get_context(),
                tool=tool,
                args=input_data,
            )

            match result:
                case "allow":
                    return PermissionResultAllow()
                case "skip":
                    return PermissionResultDeny(message="User skipped tool execution")
                case "abort_run" | "abort_chain":
                    return PermissionResultDeny(message="User aborted execution", interrupt=True)
                case _:
                    return PermissionResultDeny(message="Unknown confirmation result")

        # Default: deny if no input provider
        return PermissionResultDeny(message="No input provider configured")

    async def __aenter__(self) -> Self:
        """Connect to Claude Code."""
        from claude_agent_sdk import ClaudeSDKClient

        await super().__aenter__()
        await self._setup_toolsets()  # Setup toolsets before building opts (they add MCP servers)
        formatted_prompt = await self.sys_prompts.format_system_prompt(self)
        options = self._build_options(formatted_system_prompt=formatted_prompt)
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self.log.info("Claude Code client connected")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disconnect from Claude Code."""
        # Clean up tool bridge first
        await self._cleanup_bridge()
        if self._client:
            try:
                await self._client.disconnect()
                self.log.info("Claude Code client disconnected")
            except Exception:
                self.log.exception("Error disconnecting Claude Code client")
            self._client = None
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
    ) -> ChatMessage[TResult]:
        """Execute prompt against Claude Code.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID for the returned message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own

        Returns:
            ChatMessage containing the agent's response
        """
        final_message: ChatMessage[TResult] | None = None
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
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Stream events from Claude Code execution.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID for the final message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own

        Yields:
            RichAgentStreamEvent instances during execution
        """
        from claude_agent_sdk import (
            AssistantMessage,
            Message,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock as ToolUseBlockType,
            UserMessage,
        )
        from claude_agent_sdk.types import StreamEvent

        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider

        if not self._client:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        conversation = message_history if message_history is not None else self.conversation
        # Prepare prompts
        user_msg, processed_prompts, _original_message = await prepare_prompts(*prompts)
        # Join prompts into single string for Claude SDK
        prompt_text = " ".join(str(p) for p in processed_prompts)
        run_id = str(uuid.uuid4())
        # Emit run started
        run_started = RunStartedEvent(
            thread_id=self.conversation_id,
            run_id=run_id,
            agent_name=self.name,
        )
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, run_started)
        yield run_started
        request = ModelRequest(parts=[UserPromptPart(content=prompt_text)])
        model_messages: list[ModelResponse | ModelRequest] = [request]
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []
        pending_tool_calls: dict[str, ToolUseBlock] = {}

        try:
            await self._client.query(prompt_text)
            # Merge SDK messages with event queue for real-time tool event streaming
            async with merge_queue_into_iterator(
                self._client.receive_response(), self._event_queue
            ) as merged_events:
                async for event_or_message in merged_events:
                    # Check if it's a queued event (from tools via EventEmitter)
                    if not isinstance(event_or_message, Message):
                        # It's an event from the queue - yield it immediately
                        for handler in self.event_handler._wrapped_handlers:
                            await handler(None, event_or_message)
                        yield event_or_message
                        continue

                    message = event_or_message
                    # Process assistant messages - extract parts incrementally
                    if isinstance(message, AssistantMessage):
                        # Update model name from first assistant message
                        if message.model:
                            self._current_model = message.model
                        for block in message.content:
                            match block:
                                case TextBlock(text=text):
                                    text_chunks.append(text)
                                    current_response_parts.append(TextPart(content=text))
                                case ThinkingBlock(thinking=thinking):
                                    current_response_parts.append(ThinkingPart(content=thinking))
                                case ToolUseBlockType(id=tc_id, name=name, input=input_data):
                                    pending_tool_calls[tc_id] = block
                                    current_response_parts.append(
                                        ToolCallPart(
                                            tool_name=name, args=input_data, tool_call_id=tc_id
                                        )
                                    )
                                    # Emit ToolCallStartEvent with rich display info
                                    from agentpool.agents.claude_code_agent.converters import (
                                        derive_rich_tool_info,
                                    )

                                    rich_info = derive_rich_tool_info(name, input_data)
                                    tool_start_event = ToolCallStartEvent(
                                        tool_call_id=tc_id,
                                        tool_name=name,
                                        title=rich_info.title,
                                        kind=rich_info.kind,
                                        locations=rich_info.locations,
                                        content=rich_info.content,
                                        raw_input=input_data,
                                    )
                                    for handler in self.event_handler._wrapped_handlers:
                                        await handler(None, tool_start_event)
                                    yield tool_start_event
                                case ToolResultBlock(tool_use_id=tc_id, content=content):
                                    # Tool result received - flush response parts and add request
                                    if current_response_parts:
                                        response = ModelResponse(parts=current_response_parts)
                                        model_messages.append(response)
                                        current_response_parts = []

                                    # Get tool name from pending calls
                                    tool_use = pending_tool_calls.pop(tc_id, None)
                                    tool_name = tool_use.name if tool_use else "unknown"
                                    tool_input = tool_use.input if tool_use else {}
                                    tool_done_event = ToolCallCompleteEvent(
                                        tool_name=tool_name,
                                        tool_call_id=tc_id,
                                        tool_input=tool_input,
                                        tool_result=content,
                                        agent_name=self.name,
                                        message_id="",
                                    )
                                    for handler in self.event_handler._wrapped_handlers:
                                        await handler(None, tool_done_event)
                                    yield tool_done_event

                                    # Add tool return as ModelRequest
                                    part = ToolReturnPart(
                                        tool_name=tool_name, content=content, tool_call_id=tc_id
                                    )
                                    model_messages.append(ModelRequest(parts=[part]))

                    # Process user messages - may contain tool results
                    elif isinstance(message, UserMessage):
                        user_content = message.content
                        user_blocks = (
                            [user_content] if isinstance(user_content, str) else user_content
                        )
                        for user_block in user_blocks:
                            if isinstance(user_block, ToolResultBlock):
                                tc_id = user_block.tool_use_id
                                result_content = user_block.content

                                # Flush response parts
                                if current_response_parts:
                                    model_messages.append(
                                        ModelResponse(parts=current_response_parts)
                                    )
                                    current_response_parts = []

                                # Get tool name from pending calls
                                tool_use = pending_tool_calls.pop(tc_id, None)
                                tool_name = tool_use.name if tool_use else "unknown"
                                tool_input = tool_use.input if tool_use else {}
                                # Emit ToolCallCompleteEvent
                                tool_complete_event = ToolCallCompleteEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tc_id,
                                    tool_input=tool_input,
                                    tool_result=result_content,
                                    agent_name=self.name,
                                    message_id="",
                                )
                                for handler in self.event_handler._wrapped_handlers:
                                    await handler(None, tool_complete_event)
                                yield tool_complete_event
                                # Add tool return as ModelRequest
                                part = ToolReturnPart(
                                    tool_name=tool_name,
                                    content=result_content,
                                    tool_call_id=tc_id,
                                )
                                model_messages.append(ModelRequest(parts=[part]))

                    # Handle StreamEvent for real-time streaming
                    elif isinstance(message, StreamEvent):
                        event_data = message.event
                        event_type = event_data.get("type")
                        index = event_data.get("index", 0)

                        # Handle content_block_start events
                        if event_type == "content_block_start":
                            content_block = event_data.get("content_block", {})
                            block_type = content_block.get("type")

                            if block_type == "text":
                                start_event = PartStartEvent(index=index, part=TextPart(content=""))
                                for handler in self.event_handler._wrapped_handlers:
                                    await handler(None, start_event)
                                yield start_event

                            elif block_type == "thinking":
                                thinking_part = ThinkingPart(content="")
                                start_event = PartStartEvent(index=index, part=thinking_part)
                                for handler in self.event_handler._wrapped_handlers:
                                    await handler(None, start_event)
                                yield start_event

                            elif block_type == "tool_use":
                                # Tool use start is handled via AssistantMessage ToolUseBlock
                                pass

                        # Handle content_block_delta events (text streaming)
                        elif event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "text_delta":
                                text_delta = delta.get("text", "")
                                if text_delta:
                                    text_part = TextPartDelta(content_delta=text_delta)
                                    delta_event = PartDeltaEvent(index=index, delta=text_part)
                                    for handler in self.event_handler._wrapped_handlers:
                                        await handler(None, delta_event)
                                    yield delta_event

                            elif delta_type == "thinking_delta":
                                thinking_delta = delta.get("thinking", "")
                                if thinking_delta:
                                    delta = ThinkingPartDelta(content_delta=thinking_delta)
                                    delta_event = PartDeltaEvent(index=index, delta=delta)
                                    for handler in self.event_handler._wrapped_handlers:
                                        await handler(None, delta_event)
                                    yield delta_event

                        # Handle content_block_stop events
                        elif event_type == "content_block_stop":
                            # We don't have the full part content here, emit with empty part
                            # The actual content was accumulated via deltas
                            end_event = PartEndEvent(index=index, part=TextPart(content=""))
                            for handler in self.event_handler._wrapped_handlers:
                                await handler(None, end_event)
                            yield end_event

                        # Skip further processing for StreamEvent - don't duplicate
                        continue

                    # Convert to events and yield
                    # (skip AssistantMessage - already streamed via StreamEvent)
                    if not isinstance(message, AssistantMessage):
                        events = claude_message_to_events(
                            message,
                            agent_name=self.name,
                            pending_tool_calls={},  # Already handled above
                        )
                        for event in events:
                            for handler in self.event_handler._wrapped_handlers:
                                await handler(None, event)
                            yield event

                    # Check for result (end of response) and capture usage info
                    if isinstance(message, ResultMessage):
                        result_message = message
                        break
                else:
                    result_message = None

        except Exception as e:
            error_event = RunErrorEvent(message=str(e), run_id=run_id, agent_name=self.name)
            for handler in self.event_handler._wrapped_handlers:
                await handler(None, error_event)
            yield error_event
            raise

        # Flush any remaining response parts
        if current_response_parts:
            model_messages.append(ModelResponse(parts=current_response_parts))

        # Determine final content - use structured output if available
        final_content: TResult = (
            result_message.structured_output  # type: ignore[assignment]
            if self._output_type is not str and result_message and result_message.structured_output
            else "".join(text_chunks)
        )

        # Build cost_info from ResultMessage if available
        cost_info: TokenCost | None = None
        if result_message and result_message.usage:
            usage = result_message.usage
            run_usage = RunUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            )
            total_cost = Decimal(str(result_message.total_cost_usd or 0))
            cost_info = TokenCost(token_usage=run_usage, total_cost=total_cost)

        chat_message = ChatMessage[TResult](
            content=final_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            model_name=self.model_name,
            messages=model_messages,
            cost_info=cost_info,
            response_time=result_message.duration_ms / 1000 if result_message else None,
        )

        # Emit stream complete
        complete_event: StreamCompleteEvent[TResult] = StreamCompleteEvent(message=chat_message)
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, complete_event)
        yield complete_event
        # Record to history
        self.message_sent.emit(chat_message)
        conversation.add_chat_messages([user_msg, chat_message])

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
    ) -> AsyncIterator[ChatMessage[TResult]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially

        Yields:
            Response messages in sequence
        """
        for prompts in prompt_groups:
            response = await self.run(*prompts)
            yield response

    async def set_model(self, model: str) -> None:
        """Set the model for future requests.

        Note: This updates the model for the next query. The client
        maintains the connection, so this takes effect on the next query().

        Args:
            model: Model name to use
        """
        self._model = model
        self._current_model = model

        if self._client:
            await self._client.set_model(model)
            self.log.info("Model changed", model=model)

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode.

        Args:
            mode: Confirmation mode - "always", "never", or "per_tool"
        """
        self.tool_confirmation_mode = mode
        # Update permission mode on client if connected
        if self._client and mode == "never":
            await self._client.set_permission_mode("bypassPermissions")
        elif self._client and mode == "always":
            await self._client.set_permission_mode("default")

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        from agentpool.talk.stats import MessageStats

        return MessageStats(messages=list(self.conversation.chat_messages))


if __name__ == "__main__":
    import os

    os.environ["ANTHROPIC_API_KEY"] = ""

    async def main() -> None:
        """Demo: Basic call to Claude Code."""
        async with ClaudeCodeAgent(name="demo", event_handlers=["detailed"]) as agent:
            print("Response (streaming): ", end="", flush=True)
            async for _ in agent.run_stream("What files are in the current directory?"):
                pass

    asyncio.run(main())
