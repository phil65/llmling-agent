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
from typing import TYPE_CHECKING, Any, Self
import uuid

import anyio
from pydantic_ai import (
    FunctionToolResultEvent,
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
    ToolCallPartDelta,
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
from agentpool.agents.modes import ModeInfo
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.messaging.messages import TokenCost
from agentpool.messaging.processing import prepare_prompts
from agentpool.models.claude_code_agents import ClaudeCodeAgentConfig
from agentpool.utils.streams import FileTracker, merge_queue_into_iterator


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
    from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput
    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment
    from slashed import BaseCommand, Command, CommandContext
    from tokonomics.model_discovery.model_info import ModelInfo
    from toprompt import AnyPromptType

    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        IndividualEventHandler,
        PromptCompatible,
    )
    from agentpool.delegation import AgentPool
    from agentpool.mcp_server.tool_bridge import ToolManagerBridge
    from agentpool.messaging import MessageHistory
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
        max_budget_usd: float | None = None,
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
        commands: Sequence[BaseCommand] | None = None,
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
            max_budget_usd: Maximum budget to consume in dollars
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
            commands: Slash commands
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
            commands=commands,
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
        self._max_budget_usd = max_budget_usd or config.max_budget_usd
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

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets from config and create bridge if needed.

        Creates providers from toolset configs, adds them to the tool manager,
        and starts an MCP bridge to expose them to Claude Code via the SDK's
        native MCP support. Also converts external MCP servers to SDK format.
        """
        from agentpool.agents.claude_code_agent.converters import convert_mcp_servers_to_sdk_format
        from agentpool.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge

        # Convert external MCP servers to SDK format first
        if self._external_mcp_servers:
            external_configs = convert_mcp_servers_to_sdk_format(self._external_mcp_servers)
            self._mcp_servers.update(external_configs)
            self.log.info("External MCP servers configured", server_count=len(external_configs))

        if not self._config.toolsets:
            return

        # Create providers from toolset configs and add to tool manager
        for toolset_config in self._config.toolsets:
            provider = toolset_config.get_provider()
            self.tools.add_provider(provider)

        server_name = f"agentpool-{self.name}-tools"
        config = BridgeConfig(transport="streamable-http", server_name=server_name)
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

    def _build_hooks(self) -> dict[str, list[Any]]:
        """Build SDK hooks configuration.

        Returns:
            Dictionary mapping hook event names to HookMatcher lists
        """
        from claude_agent_sdk.types import HookMatcher

        async def on_pre_compact(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Handle PreCompact hook by emitting a text notification for auto-compaction."""
            # input_data is PreCompactHookInput when hook_event_name == "PreCompact"
            trigger = input_data.get("trigger", "auto")

            # Only show notification for auto-compaction
            # Manual compaction is triggered via slash command which handles its own UI
            if trigger == "auto":
                text = (
                    "\n\n---\n\n"
                    "📦 **Context compaction** triggered. Summarizing conversation..."
                    "\n\n---\n\n"
                )
                delta_event = PartDeltaEvent(
                    index=0,
                    delta=TextPartDelta(content_delta=text),
                )
                await self._event_queue.put(delta_event)

            return {"continue_": True}

        return {
            "PreCompact": [HookMatcher(matcher=None, hooks=[on_pre_compact])],
        }

    def _build_options(self, *, formatted_system_prompt: str | None = None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from runtime state.

        Args:
            formatted_system_prompt: Pre-formatted system prompt from SystemPrompts manager
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk.types import SystemPromptPreset

        from agentpool.agents.claude_code_agent.converters import to_output_format

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

        return ClaudeAgentOptions(
            cwd=self._cwd,
            allowed_tools=self._allowed_tools or [],
            disallowed_tools=self._disallowed_tools or [],
            system_prompt=system_prompt,
            model=self._model,
            max_turns=self._max_turns,
            max_budget_usd=self._max_budget_usd,
            max_thinking_tokens=self._max_thinking_tokens,
            permission_mode=permission_mode,
            env=self._environment or {},
            add_dirs=self._add_dir or [],  # type: ignore[arg-type]  # SDK uses list not Sequence
            tools=self._builtin_tools,
            fallback_model=self._fallback_model,
            can_use_tool=can_use_tool,
            output_format=to_output_format(self._output_type),
            mcp_servers=self._mcp_servers or {},
            include_partial_messages=True,
            hooks=self._build_hooks(),  # type: ignore[arg-type]
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

    async def populate_commands(self) -> None:
        """Populate the command store with slash commands from Claude Code.

        Fetches available commands from the connected Claude Code server
        and registers them as slashed Commands. Should be called after
        connection is established.

        Commands that are not supported or not useful for external use
        are filtered out (e.g., login, logout, context, cost).
        """
        if not self._client:
            self.log.warning("Cannot populate commands: not connected")
            return

        server_info = await self._client.get_server_info()
        if not server_info:
            self.log.warning("No server info available for command population")
            return

        commands = server_info.get("commands", [])
        if not commands:
            self.log.debug("No commands available from Claude Code server")
            return

        # Commands to skip - not useful or problematic in this context
        unsupported = {"context", "cost", "login", "logout", "release-notes", "todos"}

        for cmd_info in commands:
            name = cmd_info.get("name", "")
            if not name or name in unsupported:
                continue

            command = self._create_claude_code_command(cmd_info)
            self._command_store.register_command(command)

        self.log.info(
            "Populated command store", command_count=len(self._command_store.list_commands())
        )

    def _create_claude_code_command(self, cmd_info: dict[str, Any]) -> Command:
        """Create a slashed Command from Claude Code command info.

        Args:
            cmd_info: Command info dict with 'name', 'description', 'argumentHint'

        Returns:
            A slashed Command that executes via Claude Code
        """
        from slashed import Command

        name = cmd_info.get("name", "")
        description = cmd_info.get("description", "")
        argument_hint = cmd_info.get("argumentHint")

        # Handle MCP commands - they have " (MCP)" suffix in Claude Code
        category = "claude_code"
        if name.endswith(" (MCP)"):
            name = f"mcp:{name.replace(' (MCP)', '')}"
            category = "mcp"

        async def execute_command(
            ctx: CommandContext[Any],
            args: list[str],
            kwargs: dict[str, str],
        ) -> None:
            """Execute the Claude Code slash command."""
            import re

            from claude_agent_sdk.types import (
                AssistantMessage,
                ResultMessage,
                TextBlock,
                UserMessage,
            )

            # Build command string
            args_str = " ".join(args) if args else ""
            if kwargs:
                kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                args_str = f"{args_str} {kwargs_str}".strip()

            full_command = f"/{name} {args_str}".strip()

            # Execute via agent run - slash commands go through as prompts
            if self._client:
                await self._client.query(full_command)
                async for msg in self._client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                await ctx.print(block.text)
                    elif isinstance(msg, UserMessage):
                        # Handle local command output wrapped in XML tags
                        content = msg.content if isinstance(msg.content, str) else ""
                        # Extract content from <local-command-stdout> or <local-command-stderr>
                        match = re.search(
                            r"<local-command-(?:stdout|stderr)>(.*?)</local-command-(?:stdout|stderr)>",
                            content,
                            re.DOTALL,
                        )
                        if match:
                            await ctx.print(match.group(1))
                    elif isinstance(msg, ResultMessage):
                        if msg.result:
                            await ctx.print(msg.result)
                        if msg.is_error:
                            await ctx.print(f"Error: {msg.subtype}")

        return Command.from_raw(
            execute_command,
            name=name,
            description=description or f"Claude Code command: {name}",
            category=category,
            usage=argument_hint,
        )

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
        deps: TDeps | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Stream events from Claude Code execution.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID for the final message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own
            deps: Optional dependencies accessible via ctx.data in tools
            event_handlers: Optional event handlers for this run (overrides agent's handlers)

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

        # Reset cancellation state
        self._cancelled = False
        self._current_stream_task = asyncio.current_task()
        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider
        if not self._client:
            raise RuntimeError("Agent not initialized - use async context manager")

        conversation = message_history if message_history is not None else self.conversation
        # Use provided event handlers or fall back to agent's handlers
        if event_handlers is not None:
            from anyenv import MultiEventHandler

            from agentpool.agents.events import resolve_event_handlers

            handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(
                resolve_event_handlers(event_handlers)
            )
        else:
            handler = self.event_handler
        # Prepare prompts
        # Get parent_id from last message in history for tree structure
        last_msg_id = conversation.get_last_message_id()
        user_msg, processed_prompts, _original_message = await prepare_prompts(
            *prompts, parent_id=last_msg_id
        )
        # Get pending parts from conversation (staged content)
        pending_parts = conversation.get_pending_parts()
        # Combine pending parts with new prompts, then join into single string for Claude SDK
        all_parts = [*pending_parts, *processed_prompts]
        prompt_text = " ".join(str(p) for p in all_parts)
        run_id = str(uuid.uuid4())
        # Emit run started
        run_started = RunStartedEvent(
            thread_id=self.conversation_id,
            run_id=run_id,
            agent_name=self.name,
        )
        await handler(None, run_started)
        yield run_started
        request = ModelRequest(parts=[UserPromptPart(content=prompt_text)])
        model_messages: list[ModelResponse | ModelRequest] = [request]
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []
        pending_tool_calls: dict[str, ToolUseBlock] = {}
        # Track tool calls that already had ToolCallStartEvent emitted (via StreamEvent)
        emitted_tool_starts: set[str] = set()

        # Accumulator for streaming tool arguments
        from agentpool.agents.tool_call_accumulator import ToolCallAccumulator

        tool_accumulator = ToolCallAccumulator()

        # Track files modified during this run
        file_tracker = FileTracker()

        # Set deps on tool bridge for access during tool invocations
        # (ContextVar doesn't work because MCP server runs in a separate task)
        if self._tool_bridge:
            self._tool_bridge.current_deps = deps
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
                                    tool_call_part = ToolCallPart(
                                        tool_name=name, args=input_data, tool_call_id=tc_id
                                    )
                                    current_response_parts.append(tool_call_part)

                                    # Emit FunctionToolCallEvent (triggers UI notification)
                                    # func_tool_event = FunctionToolCallEvent(part=tool_call_part)
                                    # await handler(None, func_tool_event)
                                    # yield func_tool_event

                                    # Only emit ToolCallStartEvent if not already emitted
                                    # via streaming (emits early with partial info)
                                    if tc_id not in emitted_tool_starts:
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
                                        # Track file modifications
                                        file_tracker.process_event(tool_start_event)
                                        await handler(None, tool_start_event)
                                        yield tool_start_event
                                    else:
                                        # Already emitted early - emit update with full args
                                        from agentpool.agents.claude_code_agent.converters import (
                                            derive_rich_tool_info,
                                        )

                                        rich_info = derive_rich_tool_info(name, input_data)
                                        updated_event = ToolCallStartEvent(
                                            tool_call_id=tc_id,
                                            tool_name=name,
                                            title=rich_info.title,
                                            kind=rich_info.kind,
                                            locations=rich_info.locations,
                                            content=rich_info.content,
                                            raw_input=input_data,
                                        )
                                        # Track file modifications using derived info
                                        file_tracker.process_event(updated_event)
                                        await handler(None, updated_event)
                                        yield updated_event
                                    # Clean up from accumulator
                                    tool_accumulator.complete(tc_id)
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

                                    # Create ToolReturnPart for the result
                                    tool_return_part = ToolReturnPart(
                                        tool_name=tool_name, content=content, tool_call_id=tc_id
                                    )

                                    # Emit FunctionToolResultEvent (for session.py to complete UI)
                                    func_result_event = FunctionToolResultEvent(
                                        result=tool_return_part
                                    )
                                    await handler(None, func_result_event)
                                    yield func_result_event

                                    # Also emit ToolCallCompleteEvent for consumers that expect it
                                    tool_done_event = ToolCallCompleteEvent(
                                        tool_name=tool_name,
                                        tool_call_id=tc_id,
                                        tool_input=tool_input,
                                        tool_result=content,
                                        agent_name=self.name,
                                        message_id="",
                                    )
                                    await handler(None, tool_done_event)
                                    yield tool_done_event

                                    # Add tool return as ModelRequest
                                    model_messages.append(ModelRequest(parts=[tool_return_part]))

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

                                # Create ToolReturnPart for the result
                                tool_return_part = ToolReturnPart(
                                    tool_name=tool_name,
                                    content=result_content,
                                    tool_call_id=tc_id,
                                )

                                # Emit FunctionToolResultEvent (for session.py to complete UI)
                                func_result_event = FunctionToolResultEvent(result=tool_return_part)
                                await handler(None, func_result_event)
                                yield func_result_event

                                # Also emit ToolCallCompleteEvent for consumers that expect it
                                tool_complete_event = ToolCallCompleteEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tc_id,
                                    tool_input=tool_input,
                                    tool_result=result_content,
                                    agent_name=self.name,
                                    message_id="",
                                )
                                await handler(None, tool_complete_event)
                                yield tool_complete_event

                                # Add tool return as ModelRequest
                                model_messages.append(ModelRequest(parts=[tool_return_part]))

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
                                await handler(None, start_event)
                                yield start_event

                            elif block_type == "thinking":
                                thinking_part = ThinkingPart(content="")
                                start_event = PartStartEvent(index=index, part=thinking_part)
                                await handler(None, start_event)
                                yield start_event

                            elif block_type == "tool_use":
                                # Emit ToolCallStartEvent early (args still streaming)
                                tc_id = content_block.get("id", "")
                                tool_name = content_block.get("name", "")
                                tool_accumulator.start(tc_id, tool_name)

                                # Derive rich info with empty args for now
                                from agentpool.agents.claude_code_agent.converters import (
                                    derive_rich_tool_info,
                                )

                                rich_info = derive_rich_tool_info(tool_name, {})
                                tool_start_event = ToolCallStartEvent(
                                    tool_call_id=tc_id,
                                    tool_name=tool_name,
                                    title=rich_info.title,
                                    kind=rich_info.kind,
                                    locations=[],  # No locations yet, args not complete
                                    content=rich_info.content,
                                    raw_input={},  # Empty, will be filled when complete
                                )
                                emitted_tool_starts.add(tc_id)
                                await handler(None, tool_start_event)
                                yield tool_start_event

                        # Handle content_block_delta events (text streaming)
                        elif event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "text_delta":
                                text_delta = delta.get("text", "")
                                if text_delta:
                                    text_part = TextPartDelta(content_delta=text_delta)
                                    delta_event = PartDeltaEvent(index=index, delta=text_part)
                                    await handler(None, delta_event)
                                    yield delta_event

                            elif delta_type == "thinking_delta":
                                thinking_delta = delta.get("thinking", "")
                                if thinking_delta:
                                    thinking_part_delta = ThinkingPartDelta(
                                        content_delta=thinking_delta
                                    )
                                    delta_event = PartDeltaEvent(
                                        index=index, delta=thinking_part_delta
                                    )
                                    await handler(None, delta_event)
                                    yield delta_event

                            elif delta_type == "input_json_delta":
                                # Accumulate tool argument JSON fragments
                                partial_json = delta.get("partial_json", "")
                                if partial_json:
                                    # Find which tool call this belongs to by index
                                    # The index corresponds to the content block index
                                    for tc_id in tool_accumulator._calls:
                                        tool_accumulator.add_args(tc_id, partial_json)
                                        # Emit PartDeltaEvent with ToolCallPartDelta
                                        tool_delta = ToolCallPartDelta(
                                            tool_name_delta=None,
                                            args_delta=partial_json,
                                            tool_call_id=tc_id,
                                        )
                                        delta_event = PartDeltaEvent(index=index, delta=tool_delta)
                                        await handler(None, delta_event)
                                        yield delta_event
                                        break  # Only one tool call streams at a time

                        # Handle content_block_stop events
                        elif event_type == "content_block_stop":
                            # We don't have the full part content here, emit with empty part
                            # The actual content was accumulated via deltas
                            end_event = PartEndEvent(index=index, part=TextPart(content=""))
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
                            await handler(None, event)
                            yield event

                    # Check for result (end of response) and capture usage info
                    if isinstance(message, ResultMessage):
                        result_message = message
                        break

                    # Note: We do NOT return early on cancellation here.
                    # The SDK docs warn against using break/return to exit receive_response()
                    # early as it can cause asyncio cleanup issues. Instead, we let the
                    # interrupt() call cause the SDK to send a ResultMessage that will
                    # naturally terminate the stream via the isinstance(message, ResultMessage)
                    # check above. The _cancelled flag is checked in process_prompt() to
                    # return the correct stop reason.
                else:
                    result_message = None

        except asyncio.CancelledError:
            self.log.info("Stream cancelled via CancelledError")
            # Emit partial response on cancellation
            response_msg = ChatMessage[TResult](
                content="".join(text_chunks),  # type: ignore[arg-type]
                role="assistant",
                name=self.name,
                message_id=message_id or str(uuid.uuid4()),
                conversation_id=self.conversation_id,
                parent_id=user_msg.message_id,
                model_name=self.model_name,
                messages=model_messages,
                finish_reason="stop",
                metadata=file_tracker.get_metadata(),
            )
            complete_event = StreamCompleteEvent(message=response_msg)
            await handler(None, complete_event)
            yield complete_event
            # Record to history even on cancellation so context is preserved
            self.message_sent.emit(response_msg)
            conversation.add_chat_messages([user_msg, response_msg])
            return

        except Exception as e:
            error_event = RunErrorEvent(message=str(e), run_id=run_id, agent_name=self.name)
            await handler(None, error_event)
            yield error_event
            raise

        finally:
            # Clear deps from tool bridge
            if self._tool_bridge:
                self._tool_bridge.current_deps = None

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

        # Determine finish reason - check if we were cancelled
        chat_message = ChatMessage[TResult](
            content=final_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            parent_id=user_msg.message_id,
            model_name=self.model_name,
            messages=model_messages,
            cost_info=cost_info,
            response_time=result_message.duration_ms / 1000 if result_message else None,
            finish_reason="stop" if self._cancelled else None,
            metadata=file_tracker.get_metadata(),
        )

        # Emit stream complete
        complete_event = StreamCompleteEvent[TResult](message=chat_message)
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

    async def interrupt(self) -> None:
        """Interrupt the currently running stream.

        Sets the cancelled flag and calls the Claude SDK's native interrupt()
        method to stop the query. The stream loop checks the flag and returns
        gracefully - we don't cancel the task ourselves to avoid CancelledError
        propagation issues.
        """
        self._cancelled = True

        # Use Claude SDK's native interrupt - this causes the SDK to stop yielding
        if self._client:
            try:
                await self._client.interrupt()
                self.log.info("Claude Code client interrupted")
            except Exception:
                self.log.exception("Failed to interrupt Claude Code client")

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

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models for Claude Code agent.

        Returns a static list of Claude models (opus, sonnet, haiku) since
        Claude Code SDK only supports these models with simple IDs.

        Returns:
            List of tokonomics ModelInfo for Claude models
        """
        from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing

        # Static Claude Code models - these are the simple IDs the SDK accepts
        # Use id_override to ensure pydantic_ai_id returns simple names like "opus"
        return [
            ModelInfo(
                id="claude-opus-4-20250514",
                name="Claude Opus",
                provider="anthropic",
                description="Claude Opus - most capable model",
                context_window=200000,
                max_output_tokens=32000,
                input_modalities={"text", "image"},
                output_modalities={"text"},
                pricing=ModelPricing(
                    prompt=0.000015,  # $15 per 1M tokens
                    completion=0.000075,  # $75 per 1M tokens
                ),
                id_override="opus",  # Claude Code SDK uses simple names
            ),
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet",
                provider="anthropic",
                description="Claude Sonnet - balanced performance and speed",
                context_window=200000,
                max_output_tokens=16000,
                input_modalities={"text", "image"},
                output_modalities={"text"},
                pricing=ModelPricing(
                    prompt=0.000003,  # $3 per 1M tokens
                    completion=0.000015,  # $15 per 1M tokens
                ),
                id_override="sonnet",  # Claude Code SDK uses simple names
            ),
            ModelInfo(
                id="claude-haiku-3-5-20241022",
                name="Claude Haiku",
                provider="anthropic",
                description="Claude Haiku - fast and cost-effective",
                context_window=200000,
                max_output_tokens=8000,
                input_modalities={"text", "image"},
                output_modalities={"text"},
                pricing=ModelPricing(
                    prompt=0.0000008,  # $0.80 per 1M tokens
                    completion=0.000004,  # $4 per 1M tokens
                ),
                id_override="haiku",  # Claude Code SDK uses simple names
            ),
        ]

    def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for Claude Code agent.

        Claude Code exposes permission modes from the SDK.

        Returns:
            List with single ModeCategory for Claude Code permission modes
        """
        from agentpool.agents.modes import ModeCategory, ModeInfo

        # Get current mode - map our confirmation mode to Claude's permission mode
        current_id = self._permission_mode or "default"
        if self.tool_confirmation_mode == "never":
            current_id = "bypassPermissions"

        category_id = "permissions"
        return [
            ModeCategory(
                id=category_id,
                name="Mode",
                available_modes=[
                    ModeInfo(
                        id="default",
                        name="Default",
                        description="Require confirmation for tool usage",
                        category_id=category_id,
                    ),
                    ModeInfo(
                        id="acceptEdits",
                        name="Accept Edits",
                        description="Auto-approve file edits without confirmation",
                        category_id=category_id,
                    ),
                    ModeInfo(
                        id="plan",
                        name="Plan",
                        description="Planning mode - no tool execution",
                        category_id=category_id,
                    ),
                    ModeInfo(
                        id="bypassPermissions",
                        name="Bypass Permissions",
                        description="Skip all permission checks (use with caution)",
                        category_id=category_id,
                    ),
                ],
                current_mode_id=current_id,
            )
        ]

    async def set_mode(self, mode: ModeInfo | str, category_id: str | None = None) -> None:
        """Set a mode within a category.

        For Claude Code, this handles permission modes from the SDK.

        Args:
            mode: The mode to set - ModeInfo object or mode ID string
            category_id: Optional category ID (defaults to "permissions")

        Raises:
            ValueError: If the category or mode is unknown
        """
        # Extract mode_id and category from ModeInfo if provided
        if isinstance(mode, ModeInfo):
            mode_id = mode.id
            category_id = category_id or mode.category_id or None
        else:
            mode_id = mode

        # Default to first (and only) category
        if category_id is None:
            category_id = "permissions"

        if category_id != "permissions":
            msg = f"Unknown category: {category_id}. Only 'permissions' is supported."
            raise ValueError(msg)

        # Map mode_id to PermissionMode
        valid_modes: set[PermissionMode] = {"default", "acceptEdits", "plan", "bypassPermissions"}
        if mode_id not in valid_modes:
            msg = f"Unknown mode: {mode_id}. Available: {list(valid_modes)}"
            raise ValueError(msg)

        permission_mode: PermissionMode = mode_id  # type: ignore[assignment]
        self._permission_mode = permission_mode

        # Update tool confirmation mode based on permission mode
        if mode_id == "bypassPermissions":
            self.tool_confirmation_mode = "never"
        elif mode_id in ("default", "plan"):
            self.tool_confirmation_mode = "always"

        # Update SDK client if connected
        if self._client:
            await self._client.set_permission_mode(permission_mode)
            self.log.info("Permission mode changed", mode=mode_id)


if __name__ == "__main__":
    import os

    os.environ["ANTHROPIC_API_KEY"] = ""

    # async def main() -> None:
    #     """Demo: Basic call to Claude Code."""
    #     async with ClaudeCodeAgent(name="demo", event_handlers=["detailed"]) as agent:
    #         print("Response (streaming): ", end="", flush=True)
    #         async for _ in agent.run_stream("What files are in the current directory?"):
    #             pass

    async def main() -> None:
        """Demo: Basic call to Claude Code."""
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        options = ClaudeAgentOptions(include_partial_messages=True)
        client = ClaudeSDKClient(options=options)
        await client.connect()
        prompt = "Do one tool call. list the cwd"
        await client.query(prompt)
        async for message in client.receive_response():
            print(message)

    anyio.run(main)
