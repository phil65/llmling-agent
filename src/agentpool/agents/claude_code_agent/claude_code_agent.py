"""ClaudeCodeAgent - Native Claude Agent SDK integration.

This module provides an agent implementation that wraps the Claude Agent SDK's
ClaudeSDKClient for native integration with agentpool.

The ClaudeCodeAgent acts as a client to the Claude Code CLI, enabling:
- Bidirectional streaming communication
- Tool permission handling via callbacks
- Integration with agentpool's event system

Tool Call Event Flow
--------------------
The SDK streams events in a specific order. Understanding this is critical for
avoiding race conditions with permission dialogs:

1. **content_block_start** (StreamEvent)
   - Contains tool_use_id, tool name
   - We emit ToolCallStartEvent here (early, with empty args)
   - ACP converter sends `tool_call` notification to client

2. **content_block_delta** (StreamEvent, multiple)
   - Contains input_json_delta with partial JSON args
   - We emit PartDeltaEvent(ToolCallPartDelta) for streaming
   - ACP converter accumulates args, doesn't send notifications

3. **AssistantMessage** with ToolUseBlock
   - Contains complete tool call info (id, name, full args)
   - We do NOT emit events here (would race with permission)
   - Just track file modifications silently

4. **content_block_stop**, **message_delta**, **message_stop** (StreamEvent)
   - Signal completion of the message

5. **can_use_tool callback** (~100ms after message_stop)
   - SDK calls our permission callback
   - We send permission request to ACP client
   - Client shows permission dialog to user
   - IMPORTANT: No notifications should be sent while dialog is open!

6. **Tool execution or denial**
   - If allowed: tool runs, emits ToolCallCompleteEvent
   - If denied: SDK receives denial, continues with next turn

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
import re
from typing import TYPE_CHECKING, Any, Literal, Self
import uuid

import anyio
from pydantic_ai import (
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
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
from pydantic_ai.usage import RequestUsage

from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.claude_code_agent.converters import claude_message_to_events
from agentpool.agents.events import (
    PartStartEvent,
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from agentpool.agents.events.processors import FileTracker
from agentpool.agents.modes import ModeInfo
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.messaging.messages import TokenCost
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
    from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput
    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import UserContent
    from slashed import BaseCommand, Command, CommandContext
    from tokonomics.model_discovery.model_info import ModelInfo
    from tokonomics.model_names import AnthropicMaxModelName
    from toprompt import AnyPromptType

    from agentpool.agents.claude_code_agent.models import (
        ClaudeCodeCommandInfo,
        ClaudeCodeServerInfo,
    )
    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        IndividualEventHandler,
        MCPServerStatus,
        PromptCompatible,
    )
    from agentpool.delegation import AgentPool
    from agentpool.mcp_server.tool_bridge import ToolManagerBridge
    from agentpool.messaging import MessageHistory
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode


logger = get_logger(__name__)

# Pattern to strip MCP server prefix from tool names
# Format: mcp__agentpool-{agent_name}-tools__{tool_name}
_MCP_TOOL_PATTERN = re.compile(r"^mcp__agentpool-(.+)-tools__(.+)$")


def _strip_mcp_prefix(tool_name: str) -> str:
    """Strip MCP server prefix from tool names for cleaner UI display.

    Handles dynamic prefixes like mcp__agentpool-{agent_name}-tools__{tool}
    """
    if match := _MCP_TOOL_PATTERN.match(tool_name):
        return match.group(2)  # group(1) is agent name, group(2) is tool name
    return tool_name


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
        model: AnthropicMaxModelName | str | None = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: PermissionMode | None = None,
        mcp_servers: Sequence[MCPServerConfig] | None = None,
        environment: dict[str, str] | None = None,
        add_dir: list[str] | None = None,
        builtin_tools: list[str] | None = None,
        fallback_model: AnthropicMaxModelName | str | None = None,
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
        self._current_model: AnthropicMaxModelName | str | None = self._model
        self._sdk_session_id: str | None = None  # Session ID from Claude SDK init message
        self.deps_type = type(None)

        # ToolBridge state for exposing toolsets via MCP
        self._tool_bridge: ToolManagerBridge | None = None
        self._owns_bridge = False  # Track if we created the bridge (for cleanup)
        self._mcp_servers: dict[str, McpServerConfig] = {}  # Claude SDK MCP server configs

        # Track pending tool call for permission matching
        # Maps tool_name to tool_call_id for matching permissions to tool call UI parts
        self._pending_tool_call_ids: dict[str, str] = {}

    @classmethod
    def from_config(
        cls,
        config: ClaudeCodeAgentConfig,
        *,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        output_type: type[TResult] | None = None,
    ) -> Self:
        """Create a ClaudeCodeAgent from a config object.

        This is the preferred way to instantiate a ClaudeCodeAgent from configuration.

        Args:
            config: Claude Code agent configuration
            event_handlers: Optional event handlers (merged with config handlers)
            input_provider: Optional input provider for user interactions
            agent_pool: Optional agent pool for coordination
            output_type: Optional output type for structured output

        Returns:
            Configured ClaudeCodeAgent instance
        """
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
            tool_confirmation_mode=config.requires_tool_confirmation,
            output_type=output_type,
        )

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

        if not self._config.tools:
            return

        # Create providers from tool configs and add to tool manager
        for provider in self._config.get_tool_providers():
            self.tools.add_provider(provider)
        server_name = f"agentpool-{self.name}-tools"
        config = BridgeConfig(transport="streamable-http", server_name=server_name)
        self._tool_bridge = ToolManagerBridge(node=self, config=config)
        await self._tool_bridge.start()
        self._owns_bridge = True
        # Get Claude SDK-compatible MCP config and merge into our servers dict
        mcp_config = self._tool_bridge.get_claude_mcp_server_config()
        self._mcp_servers.update(mcp_config)
        self.log.info("Tools initialized", tool_count=len(self._config.tools))

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

    @property
    def model_name(self) -> str | None:
        """Get the model name."""
        return self._current_model

    def get_mcp_server_info(self) -> dict[str, MCPServerStatus]:
        """Get information about configured MCP servers.

        Returns a dict mapping server names to their status info. This is used
        by the OpenCode /mcp endpoint to display MCP servers in the sidebar.

        Returns:
            Dict mapping server name to MCPServerStatus dataclass
        """
        from agentpool.common_types import MCPServerStatus

        result: dict[str, MCPServerStatus] = {}
        for name, config in self._mcp_servers.items():
            server_type = config.get("type", "unknown")
            result[name] = MCPServerStatus(
                name=name,
                status="connected",  # Claude SDK manages connections
                server_type=server_type,
            )
        return result

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
            """Handle PreCompact hook by emitting a CompactionEvent."""
            from agentpool.agents.events import CompactionEvent

            # input_data is PreCompactHookInput when hook_event_name == "PreCompact"
            trigger_value = input_data.get("trigger", "auto")
            trigger: Literal["auto", "manual"] = "manual" if trigger_value == "manual" else "auto"
            # Emit semantic CompactionEvent - consumers handle display differently
            ses_id = self.conversation_id or "unknown"
            compaction_event = CompactionEvent(session_id=ses_id, trigger=trigger, phase="starting")
            await self._event_queue.put(compaction_event)
            return {"continue_": True}

        return {"PreCompact": [HookMatcher(matcher=None, hooks=[on_pre_compact])]}

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
        import uuid

        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        from agentpool.tools import FunctionTool

        # Auto-grant if confirmation mode is "never" (bypassPermissions)
        if self.tool_confirmation_mode == "never":
            return PermissionResultAllow()

        # For "acceptEdits" mode: auto-allow edit/write tools only
        if self._permission_mode == "acceptEdits":
            # Extract the actual tool name from MCP-style names
            # e.g., "mcp__agentpool-claude-tools__edit" -> "edit"
            actual_tool_name = tool_name
            if "__" in tool_name:
                actual_tool_name = tool_name.rsplit("__", 1)[-1]
            # Auto-allow file editing tools
            if actual_tool_name.lower() in ("edit", "write", "edit_file", "write_file"):
                return PermissionResultAllow()

        # For "default" mode and non-edit tools in "acceptEdits" mode:
        # Ask for confirmation via input provider
        if self._input_provider:
            # Get tool_use_id from SDK context if available (requires SDK >= 0.1.19)
            # TODO: Remove fallback once claude-agent-sdk with tool_use_id is released
            if hasattr(context, "tool_use_id") and (tc_id := context.tool_use_id):  # pyright: ignore[reportAttributeAccessIssue]
                tool_call_id = tc_id
            else:
                # Fallback: look up from streaming events or generate our own
                tool_call_id = self._pending_tool_call_ids.get(tool_name)
                if not tool_call_id:
                    tool_call_id = f"perm_{uuid.uuid4().hex[:12]}"
                    self._pending_tool_call_ids[tool_name] = tool_call_id

            display_name = _strip_mcp_prefix(tool_name)
            self.log.debug("Permission request", tool_name=display_name, tool_call_id=tool_call_id)
            # Create a dummy Tool for the confirmation dialog
            desc = f"Claude Code tool: {tool_name}"
            tool = FunctionTool(callable=lambda: None, name=display_name, description=desc)
            ctx = self.get_context()
            # Attach tool_call_id to context for permission event
            ctx.tool_call_id = tool_call_id
            # Also pass tool input for ACPInputProvider to generate proper title
            ctx.tool_input = input_data
            ctx.tool_name = tool_name
            result = await self._input_provider.get_tool_confirmation(
                context=ctx,
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
        """Connect to Claude Code with deferred client connection."""
        from claude_agent_sdk import ClaudeSDKClient

        await super().__aenter__()
        await self._setup_toolsets()  # Setup toolsets before building opts (they add MCP servers)
        formatted_prompt = await self.sys_prompts.format_system_prompt(self)
        options = self._build_options(formatted_system_prompt=formatted_prompt)
        self._client = ClaudeSDKClient(options=options)
        # Defer connection - will be awaited in run_stream/ensure_initialized
        # Note: We can't use create_task here because the SDK's internal task groups
        # must be entered and exited from the same task (anyio limitation)
        self._connect_pending = True
        return self

    async def _do_connect(self) -> None:
        """Actually connect the client. Must be called from the task that will use it."""
        if not self._connect_pending:
            return
        if not self._client:
            msg = "Client not created - call __aenter__ first"
            raise RuntimeError(msg)

        await self._client.connect()
        await self.populate_commands()
        self._connect_pending = False
        self.log.info("Claude Code client connected")

    async def ensure_initialized(self) -> None:
        """Wait for client connection to complete."""
        await self._do_connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disconnect from Claude Code."""
        # Clean up tool bridge first
        if self._tool_bridge and self._owns_bridge:
            await self._tool_bridge.stop()
        self._tool_bridge = None
        self._owns_bridge = False
        self._mcp_servers.clear()
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
        server_info = await self.get_server_info()
        if not server_info:
            self.log.warning("No server info available for command population")
            return
        if not server_info.commands:
            self.log.debug("No commands available from Claude Code server")
            return
        # Commands to skip - not useful or problematic in this context
        unsupported = {"login", "logout", "release-notes", "todos"}
        for cmd_info in server_info.commands:
            name = cmd_info.name
            if not name or name in unsupported:
                continue

            command = self._create_claude_code_command(cmd_info)
            self._command_store.register_command(command)
        command_count = len(self._command_store.list_commands())
        self.log.info("Populated command store", command_count=command_count)

    def _create_claude_code_command(self, cmd_info: ClaudeCodeCommandInfo) -> Command:
        """Create a slashed Command from Claude Code command info.

        Args:
            cmd_info: Command info dict with 'name', 'description', 'argumentHint'

        Returns:
            A slashed Command that executes via Claude Code
        """
        from slashed import Command

        name = cmd_info.name
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
            description=cmd_info.description or f"Claude Code command: {name}",
            category=category,
            usage=cmd_info.argument_hint,
        )

    async def _stream_events(  # noqa: PLR0915
        self,
        prompts: list[UserContent],
        *,
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
        from anyenv import MultiEventHandler
        from claude_agent_sdk import (
            AssistantMessage,
            Message,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock as ToolUseBlockType,
            UserMessage,
        )
        from claude_agent_sdk.types import StreamEvent

        from agentpool.agents.events import resolve_event_handlers
        from agentpool.agents.events.infer_info import derive_rich_tool_info
        from agentpool.agents.tool_call_accumulator import ToolCallAccumulator

        # Ensure client is connected (waits for deferred init if needed)
        await self.ensure_initialized()
        # Reset cancellation state
        self._cancelled = False
        self._current_stream_task = asyncio.current_task()
        # Initialize conversation_id on first run and log to storage
        # Use passed conversation_id if provided (e.g., from chained agents)
        # TODO: decide whether we should store CC sessions ourselves
        # For Claude Code, session_id comes from the SDK's init message:
        #   if hasattr(message, 'subtype') and message.subtype == 'init':
        #       session_id = message.data.get('session_id')
        # The SDK manages its own session persistence. To resume, pass:
        #   ClaudeAgentOptions(resume=session_id)
        # For now, generate our own ID to satisfy type requirements.
        # Later we could replace with SDK session_id if we decide to store CC sessions.
        if self.conversation_id is None:
            if conversation_id:
                self.conversation_id = conversation_id
            else:
                from agentpool.utils.identifiers import generate_session_id

                self.conversation_id = generate_session_id()
            # Extract text from prompts for title generation
            initial_prompt = " ".join(str(p) for p in prompts if isinstance(p, str))
            await self.log_conversation(initial_prompt or None)

        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider
        if not self._client:
            raise RuntimeError("Agent not initialized - use async context manager")

        conversation = message_history if message_history is not None else self.conversation
        # Use provided event handlers or fall back to agent's handlers
        if event_handlers is not None:
            handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(
                resolve_event_handlers(event_handlers)
            )
        else:
            handler = self.event_handler
        # Prepare prompts
        # Get parent_id from last message in history for tree structure
        # Use passed parent_id if provided, otherwise get from conversation history
        effective_parent_id = (
            parent_id if parent_id is not None else conversation.get_last_message_id()
        )
        user_msg = ChatMessage.user_prompt(message=prompts, parent_id=effective_parent_id)
        processed_prompts = prompts
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
        tool_accumulator = ToolCallAccumulator()
        # Track files modified during this run
        file_tracker = FileTracker()
        # Set deps on tool bridge for access during tool invocations
        # (ContextVar doesn't work because MCP server runs in a separate task)
        if self._tool_bridge:
            self._tool_bridge.current_deps = deps

        # Handle ephemeral execution (fork session if store_history=False)
        fork_client = None
        active_client = self._client

        if not store_history and self._sdk_session_id:
            # Create fork client that shares parent's context but has separate session ID
            # See: src/agentpool/agents/claude_code_agent/FORKING.md
            from claude_agent_sdk import ClaudeSDKClient

            # Build options using same method as main client
            fork_options = self._build_options()
            # Add fork-specific parameters
            fork_options.resume = self._sdk_session_id  # Fork from current session
            fork_options.fork_session = True  # Create new session ID

            fork_client = ClaudeSDKClient(options=fork_options)
            await fork_client.connect()
            active_client = fork_client

        try:
            await active_client.query(prompt_text)
            # Merge SDK messages with event queue for real-time tool event streaming
            async with merge_queue_into_iterator(
                active_client.receive_response(), self._event_queue
            ) as merged_events:
                async for event_or_message in merged_events:
                    # Check if it's a queued event (from tools via EventEmitter)
                    if not isinstance(event_or_message, Message):
                        # It's an event from the queue - yield it immediately
                        await handler(None, event_or_message)
                        yield event_or_message
                        continue

                    message = event_or_message
                    # Capture SDK session ID from init message
                    if isinstance(message, SystemMessage):
                        if message.subtype == "init" and "session_id" in message.data:
                            self._sdk_session_id = message.data["session_id"]
                        continue

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
                                    display_name = _strip_mcp_prefix(name)
                                    tool_call_part = ToolCallPart(
                                        tool_name=display_name, args=input_data, tool_call_id=tc_id
                                    )
                                    current_response_parts.append(tool_call_part)

                                    # Emit FunctionToolCallEvent (triggers UI notification)
                                    # func_tool_event = FunctionToolCallEvent(part=tool_call_part)
                                    # await handler(None, func_tool_event)
                                    # yield func_tool_event

                                    # Only emit ToolCallStartEvent if not already emitted
                                    # via streaming (emits early with partial info)
                                    if tc_id not in emitted_tool_starts:
                                        rich_info = derive_rich_tool_info(name, input_data)
                                        tool_start_event = ToolCallStartEvent(
                                            tool_call_id=tc_id,
                                            tool_name=display_name,
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
                                    # Already emitted ToolCallStartEvent early via streaming.
                                    # Dont emit a progress update here - it races with
                                    # permission requests and causes Zed to cancel the dialog.
                                    # Just track file modifications.
                                    elif file_path := file_tracker.extractor(
                                        display_name, input_data
                                    ):
                                        file_tracker.touched_files.add(file_path)
                                    # Clean up from accumulator (always, both branches)
                                    tool_accumulator.complete(tc_id)
                                case ToolResultBlock(tool_use_id=tc_id, content=content):
                                    # Tool result received - flush response parts and add request
                                    if current_response_parts:
                                        response = ModelResponse(parts=current_response_parts)
                                        model_messages.append(response)
                                        current_response_parts = []

                                    # Get tool name from pending calls
                                    tool_use = pending_tool_calls.pop(tc_id, None)
                                    tool_name = _strip_mcp_prefix(
                                        tool_use.name if tool_use else "unknown"
                                    )
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
                                tool_name = _strip_mcp_prefix(
                                    tool_use.name if tool_use else "unknown"
                                )
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
                                start_event = PartStartEvent.text(index=index, content="")
                                await handler(None, start_event)
                                yield start_event

                            elif block_type == "thinking":
                                start_event = PartStartEvent.thinking(index=index, content="")
                                await handler(None, start_event)
                                yield start_event

                            elif block_type == "tool_use":
                                # Emit ToolCallStartEvent early (args still streaming)
                                tc_id = content_block.get("id", "")
                                raw_tool_name = content_block.get("name", "")
                                tool_name = _strip_mcp_prefix(raw_tool_name)
                                tool_accumulator.start(tc_id, tool_name)
                                # Track for permission matching - permission callback will use this
                                # Use raw name since SDK uses raw names for permissions
                                self._pending_tool_call_ids[raw_tool_name] = tc_id
                                # Derive rich info with empty args for now
                                rich_info = derive_rich_tool_info(raw_tool_name, {})
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
            # Build metadata with file tracking and SDK session ID
            metadata = file_tracker.get_metadata()
            if self._sdk_session_id:
                metadata["sdk_session_id"] = self._sdk_session_id

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
                metadata=metadata,
            )
            complete_event = StreamCompleteEvent(message=response_msg)
            await handler(None, complete_event)
            yield complete_event
            # Record to history even on cancellation so context is preserved
            self.message_sent.emit(response_msg)
            await self.log_message(user_msg)
            await self.log_message(response_msg)
            conversation.add_chat_messages([user_msg, response_msg])
            return

        except Exception as e:
            error_event = RunErrorEvent(message=str(e), run_id=run_id, agent_name=self.name)
            await handler(None, error_event)
            yield error_event
            raise

        finally:
            # Disconnect fork client if we created one
            if fork_client:
                try:
                    await fork_client.disconnect()
                except Exception as e:  # noqa: BLE001
                    get_logger(__name__).warning(f"Error disconnecting fork client: {e}")

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

        # Build cost_info and usage from ResultMessage if available
        cost_info: TokenCost | None = None
        request_usage: RequestUsage | None = None
        if result_message and result_message.usage:
            usage_dict = result_message.usage
            run_usage = RunUsage(
                input_tokens=usage_dict.get("input_tokens", 0),
                output_tokens=usage_dict.get("output_tokens", 0),
                cache_read_tokens=usage_dict.get("cache_read_input_tokens", 0),
                cache_write_tokens=usage_dict.get("cache_creation_input_tokens", 0),
            )
            total_cost = Decimal(str(result_message.total_cost_usd or 0))
            cost_info = TokenCost(token_usage=run_usage, total_cost=total_cost)
            # Also set usage for OpenCode compatibility
            request_usage = RequestUsage(
                input_tokens=usage_dict.get("input_tokens", 0),
                output_tokens=usage_dict.get("output_tokens", 0),
                cache_read_tokens=usage_dict.get("cache_read_input_tokens", 0),
                cache_write_tokens=usage_dict.get("cache_creation_input_tokens", 0),
            )

        # Determine finish reason - check if we were cancelled
        # Build metadata with file tracking and SDK session ID
        metadata = file_tracker.get_metadata()
        if self._sdk_session_id:
            metadata["sdk_session_id"] = self._sdk_session_id

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
            usage=request_usage or RequestUsage(),
            response_time=result_message.duration_ms / 1000 if result_message else None,
            finish_reason="stop" if self._cancelled else None,
            metadata=metadata,
        )

        # Emit stream complete
        complete_event = StreamCompleteEvent[TResult](message=chat_message)
        await handler(None, complete_event)
        yield complete_event
        # Record to history
        self.message_sent.emit(chat_message)
        await self.log_message(user_msg)
        await self.log_message(chat_message)
        conversation.add_chat_messages([user_msg, chat_message])
        await self.connections.route_message(chat_message, wait=wait_for_connections)

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

    async def set_model(self, model: AnthropicMaxModelName | str) -> None:
        """Set the model for future requests.

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
        # Map confirmation mode to permission mode
        if mode == "never":
            self._permission_mode = "bypassPermissions"
        elif mode in {"always", "per_tool"}:
            self._permission_mode = "default"
        # Update permission mode on client if connected
        if self._client and self._permission_mode:
            await self._client.set_permission_mode(self._permission_mode)

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models for Claude Code agent.

        Returns a static list of Claude models (opus, sonnet, haiku) since
        Claude Code SDK only supports these models with simple IDs.

        Returns:
            List of tokonomics ModelInfo for Claude models
        """
        from agentpool.agents.claude_code_agent.static_info import MODELS

        return MODELS

    async def get_server_info(self) -> ClaudeCodeServerInfo | None:
        """Get server initialization info from Claude Code.

        Returns information from the Claude Code server including:
        - Available models (opus, sonnet, haiku) with descriptions and pricing
        - Available slash commands with descriptions and argument hints
        - Current and available output styles
        - Account information (token source, API key source)
        """
        from agentpool.agents.claude_code_agent.models import ClaudeCodeServerInfo

        if not self._client:
            self.log.warning("Cannot get server info: not connected")
            return None
        # Get raw server info from SDK client
        raw_info = await self._client.get_server_info()
        if not raw_info:
            self.log.warning("No server info available from Claude Code")
            return None
        return ClaudeCodeServerInfo.model_validate(raw_info)

    async def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for Claude Code agent.

        Claude Code exposes permission modes and model selection.

        Returns:
            List of ModeCategory for permissions and models
        """
        from agentpool.agents.claude_code_agent.static_info import MODES
        from agentpool.agents.modes import ModeCategory, ModeInfo

        categories: list[ModeCategory] = []
        # Permission modes
        current_id = self._permission_mode or "default"
        if self.tool_confirmation_mode == "never":
            current_id = "bypassPermissions"

        categories.append(
            ModeCategory(
                id="permissions",
                name="Mode",
                available_modes=MODES,
                current_mode_id=current_id,
                category="mode",
            )
        )

        # Model selection
        models = await self.get_available_models()
        if models:
            current_model = self.model_name or (models[0].id if models else "")
            modes = [
                ModeInfo(
                    id=m.id,
                    name=m.name or m.id,
                    description=m.description or "",
                    category_id="model",
                )
                for m in models
            ]
            categories.append(
                ModeCategory(
                    id="model",
                    name="Model",
                    available_modes=modes,
                    current_mode_id=current_model,
                    category="model",
                )
            )

        return categories

    async def set_mode(self, mode: ModeInfo | str, category_id: str | None = None) -> None:
        """Set a mode within a category.

        For Claude Code, this handles:
        - "permissions" category: permission modes from the SDK
        - "model" category: model selection

        Args:
            mode: The mode to set - ModeInfo object or mode ID string
            category_id: Category ID ("permissions" or "model")

        Raises:
            ValueError: If the category or mode is unknown
        """
        from agentpool.agents.claude_code_agent.static_info import VALID_MODES

        # Extract mode_id and category from ModeInfo if provided
        if isinstance(mode, ModeInfo):
            mode_id = mode.id
            category_id = category_id or mode.category_id
        else:
            mode_id = mode

        # Default to permissions if no category specified
        if category_id is None:
            category_id = "permissions"

        if category_id == "permissions":
            # Map mode_id to PermissionMode
            if mode_id not in VALID_MODES:
                msg = f"Unknown permission mode: {mode_id}. Available: {list(VALID_MODES)}"
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

        elif category_id == "model":
            # Validate model exists
            models = await self.get_available_models()
            if models:
                valid_ids = {m.id for m in models}
                if mode_id not in valid_ids:
                    msg = f"Unknown model: {mode_id}. Available: {valid_ids}"
                    raise ValueError(msg)
            # Set the model using set_model method
            await self.set_model(mode_id)
            self.log.info("Model changed", model=mode_id)

        else:
            msg = f"Unknown category: {category_id}. Available: permissions, model"
            raise ValueError(msg)


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
