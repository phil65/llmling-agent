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
        env="/path/to/project",
        allowed_tools=["Read", "Write", "Bash"],
    ) as agent:
        async for event in agent.run_stream("Write a hello world program"):
            print(event)
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
from decimal import Decimal
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self
import uuid

import anyio
from pydantic_ai import (
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
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
from agentpool.agents.claude_code_agent.converters import (
    claude_message_to_events,
    confirmation_result_to_native,
    convert_mcp_servers_to_sdk_format,
    convert_to_opencode_metadata,
    to_claude_system_prompt,
    to_output_format,
)
from agentpool.agents.claude_code_agent.exceptions import raise_if_usage_limit_reached
from agentpool.agents.claude_code_agent.static_info import models_to_category
from agentpool.agents.events import (
    PartDeltaEvent,
    PartStartEvent,
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    ToolResultMetadataEvent,
)
from agentpool.agents.events.infer_info import derive_rich_tool_info
from agentpool.agents.events.processors import FileTracker
from agentpool.agents.exceptions import (
    AgentNotInitializedError,
    UnknownCategoryError,
    UnknownModeError,
)
from agentpool.agents.tool_call_accumulator import ToolCallAccumulator
from agentpool.common_types import MCPServerStatus
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.messaging.messages import TokenCost
from agentpool.sessions.models import SessionData
from agentpool.utils.streams import merge_queue_into_iterator
from agentpool.utils.time_utils import get_now, parse_iso_timestamp


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from clawd_code_sdk import (
        AgentDefinition,
        ClaudeSDKClient,
        McpServerConfig,
        PermissionMode,
        PermissionResult,
        ToolPermissionContext,
        ToolUseBlock,
        UserMessage,
    )
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
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import AnyEventHandlerType, StrPath
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageHistory
    from agentpool.models.claude_code_agents import ClaudeCodeAgentConfig, SettingSource
    from agentpool.resource_providers import ResourceProvider
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)

ThinkingMode = Literal["off", "4k", "8k", "16k", "32k"]

_MCP_TOOL_PATTERN = re.compile(r"^mcp__agentpool-(.+)-tools__(.+)$")
"""Pattern to detect CC-provided tool names ( mcp__agentpool-{agent_name}-tools__{tool_name} )."""

EXCLUDED_SLASH_COMMANDS = frozenset({"login", "logout", "release-notes", "todos"})
"""Slash commands that don't make sense when Claude Code is used as a sub-agent"""

THINKING_MODE_TOKENS: dict[ThinkingMode, int] = {
    "off": 0,
    "4k": 4000,
    "8k": 8000,
    "16k": 16000,
    "32k": 32000,
}
"""Token limit for each thinking mode."""


def _strip_mcp_prefix(tool_name: str) -> str:
    """Strip MCP server prefix from tool names for cleaner UI display.

    Handles dynamic prefixes like mcp__agentpool-{agent_name}-tools__{tool}
    """
    if match := _MCP_TOOL_PATTERN.match(tool_name):
        return match.group(2)  # group(1) is agent name, group(2) is tool name
    return tool_name


def parse_command_output(msg: UserMessage) -> str | None:
    content = msg.content if isinstance(msg.content, str) else ""
    # Extract content from <local-command-stdout> or <local-command-stderr>
    pattern = r"<local-command-(?:stdout|stderr)>(.*?)</local-command-(?:stdout|stderr)>"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1) if match else None


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

    AGENT_TYPE: ClassVar = "claude"

    def __init__(
        self,
        *,
        name: str | None = None,
        deps_type: type[TDeps] | None = None,
        description: str | None = None,
        display_name: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        system_prompt: str | Sequence[str | AnyPromptType] | None = None,
        include_builtin_system_prompt: bool = True,
        model: AnthropicMaxModelName | str | None = "opus",
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: PermissionMode | None = None,
        mcp_servers: Sequence[MCPServerConfig] | None = None,
        env_vars: dict[str, str] | None = None,
        add_dir: list[str] | None = None,
        builtin_tools: list[str] | None = None,
        fallback_model: AnthropicMaxModelName | str | None = None,
        dangerously_skip_permissions: bool = False,
        setting_sources: list[SettingSource] | None = None,
        use_subscription: bool = False,
        env: ExecutionEnvironment | StrPath | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        output_type: type[TResult] | None = None,
        builtin_subagents: dict[str, AgentDefinition] | None = None,
        commands: Sequence[BaseCommand] | None = None,
        hooks: AgentHooks | None = None,
        session_id: str | None = None,
        toolsets: list[ResourceProvider] | None = None,
    ) -> None:
        """Initialize ClaudeCodeAgent.

        Args:
            name: Agent name
            deps_type: Type of dependencies for the agent
            description: Agent description
            display_name: Display name for UI
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
            env_vars: Environment variables for the agent process
            add_dir: Additional directories to allow tool access to
            builtin_tools: Available tools from built-in set. Special: "LSP" for code intelligence,
                           "Chrome" for browser control
            fallback_model: Fallback model when default is overloaded
            dangerously_skip_permissions: Bypass all permission checks (sandboxed only)
            setting_sources: Setting sources to load ("user", "project", "local")
            use_subscription: Force Claude subscription usage instead of API key
            env: Execution environment
            input_provider: Provider for user input/confirmations
            agent_pool: Agent pool for multi-agent coordination
            enable_logging: Whether to enable logging
            event_configs: Event configuration
            event_handlers: Event handlers for streaming events
            output_type: Type for structured output (uses JSON schema)
            builtin_subagents: builtin Subagents configuration
            commands: Slash commands
            hooks: Lifecycle hooks for intercepting agent behavior
            session_id: Session ID to resume on connect (avoids reconnect overhead)
            toolsets: Resource providers for tools to expose via MCP bridge
        """
        from agentpool.agents.claude_code_agent.hook_manager import ClaudeCodeHookManager
        from agentpool.agents.sys_prompts import SystemPrompts
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge
        from agentpool_config.storage import ClaudeStorageConfig
        from agentpool_storage.claude_provider import ClaudeStorageProvider

        super().__init__(
            name=name or "claude_code",
            description=description,
            deps_type=deps_type,
            display_name=display_name,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs,
            env=env,
            input_provider=input_provider,
            output_type=output_type or str,  # type: ignore[arg-type]
            event_handlers=event_handlers,
            commands=commands,
            hooks=hooks,
        )
        self._subagents = builtin_subagents
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._include_builtin_system_prompt = include_builtin_system_prompt
        # Initialize SystemPrompts manager
        all_prompts: list[AnyPromptType] = []
        if system_prompt is not None:
            if isinstance(system_prompt, str):
                all_prompts.append(system_prompt)
            else:
                all_prompts.extend(system_prompt)
        prompt_manager = agent_pool.prompt_manager if agent_pool else None
        self.sys_prompts = SystemPrompts(all_prompts, prompt_manager=prompt_manager)
        self._model = model
        self._max_turns = max_turns
        self._max_budget_usd = max_budget_usd
        self._max_thinking_tokens = max_thinking_tokens
        self._permission_mode: PermissionMode | None = permission_mode
        self._thinking_mode: ThinkingMode = "off"
        self._external_mcp_servers = list(mcp_servers) if mcp_servers else []
        self._env_vars = env_vars
        self._add_dir = add_dir
        self._builtin_tools = builtin_tools
        self._fallback_model = fallback_model
        self._dangerously_skip_permissions = dangerously_skip_permissions
        self._setting_sources = setting_sources
        self._use_subscription = use_subscription
        self._toolsets = toolsets or []
        # Client state
        self._client: ClaudeSDKClient | None = None
        self._connection_task: asyncio.Task[None] | None = None
        self._sdk_session_id: str | None = session_id
        # ToolBridge state for exposing toolsets via MCP
        self._tool_bridge = ToolManagerBridge(node=self, injection_manager=self._injection_manager)
        self._mcp_servers: dict[str, McpServerConfig] = {}  # Claude SDK MCP server configs
        # Track pending tool call for permission matching
        self._pending_tool_call_ids: dict[str, str] = {}
        # Create Claude storage provider for session management
        claude_config = ClaudeStorageConfig(path="~/.claude")
        self._claude_storage = ClaudeStorageProvider(claude_config)
        self._hook_manager = ClaudeCodeHookManager(
            agent_name=self.name,
            agent_hooks=hooks,
            event_queue=self._event_queue,
            get_session_id=lambda: self.session_id,
            injection_manager=self._injection_manager,
        )

    @classmethod
    def from_config(
        cls,
        config: ClaudeCodeAgentConfig,
        *,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        deps_type: type[TDeps] | None = None,
    ) -> Self:
        """Create a ClaudeCodeAgent from a config object.

        All config values are extracted here and passed to the constructor.
        """
        from agentpool.models.manifest import AgentsManifest
        from agentpool.utils.result_utils import to_type

        # Get manifest from pool or create empty one
        manifest = agent_pool.manifest if agent_pool is not None else AgentsManifest()
        # Resolve output type from config
        resolved_output_type: type | None = None
        if config.output_type:
            resolved_output_type = to_type(config.output_type, manifest.responses)
        # Merge config-level handlers with provided handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[AnyEventHandlerType] = [*config_handlers, *(event_handlers or [])]
        return cls(
            # Identity
            name=config.name,
            description=config.description,
            deps_type=deps_type,
            display_name=config.display_name,
            # Claude Code settings
            allowed_tools=config.allowed_tools,
            disallowed_tools=config.disallowed_tools,
            system_prompt=config.system_prompt,
            env=config.get_execution_environment(),
            include_builtin_system_prompt=config.include_builtin_system_prompt,
            model=config.model,
            max_turns=config.max_turns,
            max_budget_usd=config.max_budget_usd,
            max_thinking_tokens=config.max_thinking_tokens,
            permission_mode=config.permission_mode,
            mcp_servers=config.get_mcp_servers(),
            env_vars=config.env_vars,
            add_dir=config.add_dir,
            builtin_tools=config.builtin_tools,
            fallback_model=config.fallback_model,
            dangerously_skip_permissions=config.dangerously_skip_permissions,
            setting_sources=config.setting_sources,
            use_subscription=config.use_subscription,
            # Toolsets
            toolsets=config.get_tool_providers() if config.tools else [],
            # Runtime
            event_configs=list(config.triggers),
            event_handlers=merged_handlers or None,
            input_provider=input_provider,
            agent_pool=agent_pool,
            output_type=resolved_output_type,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
        )

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets from config and create bridge if needed.

        Creates providers from toolset configs, adds them to the tool manager,
        and starts an MCP bridge to expose them to Claude Code via the SDK's
        native MCP support. Also converts external MCP servers to SDK format.
        """
        # Convert external MCP servers to SDK format first
        if self._external_mcp_servers:
            external_configs = convert_mcp_servers_to_sdk_format(self._external_mcp_servers)
            self._mcp_servers.update(external_configs)
            self.log.info("External MCP servers configured", server_count=len(external_configs))

        if not self._toolsets:
            return
        # Add toolset providers to tool manager
        for provider in self._toolsets:
            self.tools.add_provider(provider)
        await self._tool_bridge.start()
        # Get Claude SDK-compatible MCP config and merge into our servers dict
        mcp_config = self._tool_bridge.get_claude_mcp_server_config()
        self._mcp_servers.update(mcp_config)
        self.log.info("Toolsets initialized", toolset_count=len(self._toolsets))

    @property
    def model_name(self) -> str | None:
        """Get the requested model name."""
        return self._model

    async def get_mcp_server_info(self) -> dict[str, MCPServerStatus]:
        """Get information about configured MCP servers.

        Returns a dict mapping server names to their status info. This is used
        by the OpenCode /mcp endpoint to display MCP servers in the sidebar.

        If a client is connected, queries live status from Claude Code.
        Otherwise falls back to reporting from config.

        Returns:
            Dict mapping server name to MCPServerStatus dataclass
        """
        result: dict[str, MCPServerStatus] = {}
        # Try live status from connected client
        if self._client:
            try:
                await self.ensure_initialized()
                live_status = await self._client.get_mcp_status()
            except Exception:  # noqa: BLE001
                pass
            else:
                for server in live_status.get("mcpServers", []):
                    name = server.get("name", "unknown")
                    status = server.get("status", "disconnected")
                    server_info = server.get("serverInfo") or {}
                    result[name] = MCPServerStatus(
                        name=name,
                        status=status,
                        server_type=server.get("type", "unknown"),
                        server_name=server_info.get("name"),
                        server_version=server_info.get("version"),
                    )
                return result
        # Fallback: report from config
        for name, config in self._mcp_servers.items():
            server_type = config.get("type", "unknown")
            result[name] = MCPServerStatus(name=name, status="connected", server_type=server_type)
        return result

    def _get_client(
        self,
        *,
        system_prompt: str | None = None,
        fork_session: bool = False,
    ) -> ClaudeSDKClient:
        """Build ClaudeAgentOptions from runtime state.

        Args:
            system_prompt: Pre-formatted system prompt from SystemPrompts manager
            fork_session: Whether to fork the session
        """
        from clawd_code_sdk import ClaudeAgentOptions, ClaudeSDKClient

        sys_prompt = to_claude_system_prompt(system_prompt) if system_prompt else None
        # Determine effective permission mode
        permission_mode = self._permission_mode
        if self._dangerously_skip_permissions and not permission_mode:
            permission_mode = "bypassPermissions"
        # Determine can_use_tool callback
        bypass = permission_mode == "bypassPermissions" or self._dangerously_skip_permissions
        can_use_tool = (
            self._can_use_tool
            if self._permission_mode != "bypassPermissions" and not bypass
            else None
        )
        # Check builtin_tools for special tools that need extra handling
        builtin_tools = self._builtin_tools or []
        # Build extra_args for CLI flags not directly exposed
        extra_args: dict[str, str | None] = {}
        if "Chrome" in builtin_tools:
            extra_args["chrome"] = None
        # Build environment variables
        env = dict(self._env_vars or {})
        env["clawd_code_sdk_SKIP_VERSION_CHECK"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        if "LSP" in builtin_tools:
            # Enable LSP tool support
            env["ENABLE_LSP_TOOL"] = "1"
        if self._use_subscription:
            # Force subscription usage by clearing API key
            env["ANTHROPIC_API_KEY"] = ""

        opts = ClaudeAgentOptions(
            cwd=self.env.cwd,
            allowed_tools=self._allowed_tools or [],
            disallowed_tools=self._disallowed_tools or [],
            system_prompt=sys_prompt,
            model=self._model,
            max_turns=self._max_turns,
            max_budget_usd=self._max_budget_usd,
            max_thinking_tokens=self._max_thinking_tokens,
            permission_mode=permission_mode,
            env=env,
            agents=self._subagents,
            add_dirs=self._add_dir or [],  # type: ignore[arg-type]
            tools=self._builtin_tools,
            fallback_model=self._fallback_model,
            can_use_tool=can_use_tool,
            output_format=to_output_format(self._output_type),
            mcp_servers=self._mcp_servers or {},
            include_partial_messages=True,
            hooks=self._hook_manager.build_hooks(),  # type: ignore[arg-type]
            setting_sources=self._setting_sources,
            extra_args=extra_args,
            resume=self._sdk_session_id,
            fork_session=fork_session,
        )
        return ClaudeSDKClient(opts)

    async def _can_use_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResult:
        """Handle tool permission requests.

        This callback fires in two cases:
        1. Tool needs approval: Claude wants to use a tool that isn't auto-approved
        2. Claude asks a question: Claude calls the AskUserQuestion tool for clarification

        Args:
            tool_name: Name of the tool being called (e.g., "Bash", "Write", "AskUserQuestion")
            input_data: Tool input arguments
            context: Permission context with suggestions

        Returns:
            PermissionResult indicating allow or deny
        """
        from clawd_code_sdk import PermissionResultAllow, PermissionResultDeny

        from agentpool.agents.claude_code_agent.elicitation import handle_clarifying_questions

        # Handle AskUserQuestion specially - this is Claude asking for clarification
        if tool_name == "AskUserQuestion":
            agent_ctx = self.get_context()
            return await handle_clarifying_questions(agent_ctx, input_data, context)
        # Auto-grant if bypassPermissions mode is active
        if self._permission_mode == "bypassPermissions":
            return PermissionResultAllow()
        # Plan mode: auto-deny all tool executions (planning only, no execution)
        if self._permission_mode == "plan":
            return PermissionResultDeny(message="Plan mode active - tool execution disabled")
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
            if tc_id := context.tool_use_id:  # pyright: ignore[reportAttributeAccessIssue]
                tool_call_id: str | None = tc_id
            else:
                # Fallback: look up from streaming events or generate our own
                tool_call_id = self._pending_tool_call_ids.get(tool_name)
                if not tool_call_id:
                    tool_call_id = f"perm_{uuid.uuid4().hex[:12]}"
                    self._pending_tool_call_ids[tool_name] = tool_call_id

            display_name = _strip_mcp_prefix(tool_name)
            self.log.debug("Permission request", tool_name=display_name, tool_call_id=tool_call_id)
            ctx = self.get_context(
                tool_call_id=tool_call_id, tool_input=input_data, tool_name=tool_name
            )
            result = await self._input_provider.get_tool_confirmation(
                context=ctx,
                tool_name=display_name,
                tool_description=f"Claude Code tool: {tool_name}",
                args=input_data,
            )
            return confirmation_result_to_native(result)
        # Default: deny if no input provider
        return PermissionResultDeny(message="No input provider configured")

    async def __aenter__(self) -> Self:
        """Connect to Claude Code with deferred client connection."""
        await super().__aenter__()
        await self._setup_toolsets()  # Setup toolsets before building opts (they add MCP servers)
        formatted_prompt = await self.sys_prompts.format_system_prompt(self)
        self._client = self._get_client(system_prompt=formatted_prompt)
        # Start connection in background task to reduce first-prompt latency
        # The task owns the anyio context, we just await it when needed
        self._connection_task = asyncio.create_task(self._do_connect())
        return self

    async def _do_connect(self) -> None:
        """Actually connect the client. Runs in background task."""
        if not self._client:
            raise AgentNotInitializedError

        try:
            await self._client.connect()
            await self.populate_commands()
            self.log.info("Claude Code client connected")
        except Exception:
            self.log.exception("Failed to connect Claude Code client")
            raise

    async def reconnect(self, *, resume_session: bool = True) -> None:
        """Reconnect to Claude Code SDK, optionally resuming the current session.

        This is useful for recovering from hangs or connection issues without
        losing conversation history.

        Args:
            resume_session: If True, attempt to resume the current session using
                the stored session ID. If False, start a fresh session.
        """
        # Recreate client with new options
        session_to_resume = self._sdk_session_id if resume_session else None
        self.log.info("Reconnecting CC agent", resume=resume_session, session_id=session_to_resume)
        # Cancel existing connection if active
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connection_task
        self._connection_task = None
        # # Clean up tool bridge
        # if self._tool_bridge._mcp is not None:
        #     await self._tool_bridge.stop()
        # self._mcp_servers.clear()
        if self._client:
            try:
                await self._client.disconnect()
                self.log.info("Disconnected existing Claude Code client")
            except Exception:
                self.log.exception("Error disconnecting Claude Code client during reconnect")
            self._client = None

        # Clear session ID if not resuming (before _get_client which uses it)
        if not resume_session:
            self._sdk_session_id = None

        formatted_prompt = await self.sys_prompts.format_system_prompt(self)
        # _get_client includes resume=self._sdk_session_id automatically
        if session_to_resume:
            self.log.info("Attempting to resume session", session=session_to_resume)
        self._client = self._get_client(system_prompt=formatted_prompt)
        try:  # Reconnect in background
            self._connection_task = asyncio.create_task(self._do_connect())
            await self._connection_task
            mode = "resumed" if session_to_resume else "fresh"
            self.log.info("Claude Code agent reconnected successfully", session_mode=mode)
        except Exception:
            self.log.exception("Error reconnecting Claude Code agent")
            raise

    async def ensure_initialized(self) -> None:
        """Wait for background connection task to complete."""
        if self._connection_task and self._connection_task is not asyncio.current_task():
            await self._connection_task
            self._connection_task = None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disconnect from Claude Code."""
        # Cancel connection task if still running
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connection_task
        self._connection_task = None

        # Clean up tool bridge first
        # Only stop bridge if it was started (has _mcp set)
        if self._tool_bridge._mcp is not None:
            await self._tool_bridge.stop()
        self._mcp_servers.clear()
        if self._client:
            try:
                await self._client.disconnect()
                self.log.info("Claude Code client disconnected")
            except Exception:  # noqa: BLE001
                self.log.warning("Error disconnecting Claude Code client")
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
        # Commands to skip - not useful or problematic in this context
        commands = [
            self._create_claude_code_command(cmd_info)
            for cmd_info in server_info.commands
            if cmd_info.name and cmd_info.name not in EXCLUDED_SLASH_COMMANDS
        ]
        for command in commands:
            self._command_store.register_command(command, replace=True)
        self.log.info("Populated command store", command_count=len(commands))

    def _create_claude_code_command(self, cmd_info: ClaudeCodeCommandInfo) -> Command:
        """Create a slashed Command from Claude Code command info.

        Args:
            cmd_info: Command info dict with 'name', 'description', 'argumentHint'

        Returns:
            A slashed Command that executes via Claude Code
        """
        from clawd_code_sdk.types import AssistantMessage, ResultMessage, TextBlock, UserMessage
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
            # Build command string
            args_str = " ".join(args) if args else ""
            if kwargs:
                kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                args_str = f"{args_str} {kwargs_str}".strip()
            # Execute via agent run - slash commands go through as prompts
            if not self._client:
                return
            await self._client.query(f"/{name} {args_str}".strip())
            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            await ctx.print(block.text)
                elif isinstance(msg, UserMessage):
                    if parsed := parse_command_output(msg):
                        await ctx.print(parsed)
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
        user_msg: ChatMessage[Any],
        message_history: MessageHistory,
        effective_parent_id: str | None,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        from clawd_code_sdk import (
            AssistantMessage,
            Message,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        from clawd_code_sdk.types import StreamEvent

        await self.ensure_initialized()
        # Initialize session_id on first run and log to storage
        # Use passed session_id if provided (e.g., from chained agents)
        # TODO: decide whether we should store CC sessions ourselves
        # For Claude Code, session_id comes from the SDK's init message:
        #   if hasattr(message, 'subtype') and message.subtype == 'init':
        #       session_id = message.data.get('session_id')
        # The SDK manages its own session persistence. To resume, pass:
        #   ClaudeAgentOptions(resume=session_id)
        # Conversation ID initialization handled by BaseAgent

        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider
        if not self._client:
            raise AgentNotInitializedError
        # Get pending parts from conversation (staged content)
        # Combine pending parts with new prompts, then join into single string for Claude SDK
        #
        prompt_text = " ".join(str(p) for p in prompts)
        run_id = str(uuid.uuid4())
        # Emit run started
        assert self.session_id is not None  # Initialized by BaseAgent.run_stream()
        yield RunStartedEvent(session_id=self.session_id, run_id=run_id, agent_name=self.name)
        request = ModelRequest(parts=[UserPromptPart(content=prompt_text)])
        model_messages: list[ModelResponse | ModelRequest] = [request]
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        pending_tool_calls: dict[str, ToolUseBlock] = {}
        # Track tool calls that already had ToolCallStartEvent emitted (via StreamEvent)
        emitted_tool_starts: set[str] = set()
        tool_accumulator = ToolCallAccumulator()
        resolved_model: str | None = None
        # Track files modified during this run
        file_tracker = FileTracker()
        # Accumulate metadata events by tool_call_id (workaround for SDK stripping _meta)
        tool_metadata: dict[str, dict[str, Any]] = {}
        # Handle ephemeral execution (fork session if store_history=False)
        fork_client = None
        client = self._client
        result_message: ResultMessage | None = None

        if not store_history and self._sdk_session_id:
            # Create fork client that shares parent's context but has separate session ID
            # See: src/agentpool/agents/claude_code_agent/FORKING.md
            # Build options using same method as main client
            fork_client = self._get_client(fork_session=True)
            await fork_client.connect()
            client = fork_client

        # Set deps/input_provider on tool bridge (ContextVar doesn't work - separate task)
        try:
            await client.query(prompt_text)
            # Merge SDK messages with event queue for real-time tool event streaming
            async with (
                self._tool_bridge.set_run_context(deps, input_provider, prompt=prompts),
                merge_queue_into_iterator(client.receive_response(), self._event_queue) as events,
            ):
                async for event_or_message in events:
                    # Check if it's a queued event (from tools via EventEmitter)
                    if not isinstance(event_or_message, Message):
                        # Capture metadata events for correlation with tool results
                        if isinstance(event_or_message, ToolResultMetadataEvent):
                            tool_metadata[event_or_message.tool_call_id] = event_or_message.metadata
                            # Don't yield metadata events - they're internal correlation only
                            continue
                        # It's an event from the queue - yield it immediately
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
                        # Track resolved model from provider response
                        if message.model:
                            resolved_model = message.model
                        # Check for usage limit error
                        raise_if_usage_limit_reached(message)
                        for block in message.content:
                            match block:
                                case TextBlock(text=text):
                                    current_response_parts.append(TextPart(content=text))
                                case ThinkingBlock(thinking=thinking):
                                    current_response_parts.append(ThinkingPart(content=thinking))
                                case ToolUseBlock(id=tc_id, name=name, input=input_data):
                                    pending_tool_calls[tc_id] = block
                                    display_name = _strip_mcp_prefix(name)
                                    tool_call_part = ToolCallPart(
                                        tool_name=display_name, args=input_data, tool_call_id=tc_id
                                    )
                                    current_response_parts.append(tool_call_part)
                                    # Emit FunctionToolCallEvent (triggers UI notification)
                                    # func_tool_event = FunctionToolCallEvent(part=tool_call_part)
                                    # await event_handlers(None, func_tool_event)
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
                                        yield tool_start_event
                                    # Already emitted ToolCallStartEvent early via streaming.
                                    # Dont emit a progress update here - it races with
                                    # permission requests and causes Zed to cancel the dialog.
                                    # Just track file modifications.
                                    elif fpath := file_tracker.extractor(display_name, input_data):
                                        file_tracker.touched_files.add(fpath)
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
                                    return_part = ToolReturnPart(
                                        tool_name=tool_name, content=content, tool_call_id=tc_id
                                    )
                                    # Emit FunctionToolResultEvent (for session.py to complete UI)
                                    yield FunctionToolResultEvent(result=return_part)
                                    # Also emit ToolCallCompleteEvent for consumers that expect it
                                    yield ToolCallCompleteEvent(
                                        tool_name=tool_name,
                                        tool_call_id=tc_id,
                                        tool_input=tool_input,
                                        tool_result=content,
                                        agent_name=self.name,
                                        message_id="",
                                        metadata=tool_metadata.get(tc_id),
                                    )
                                    # Add tool return as ModelRequest
                                    model_messages.append(ModelRequest(parts=[return_part]))

                    # Process user messages - may contain tool results
                    elif isinstance(message, UserMessage):
                        user_content = message.content
                        user_blocks = (
                            [user_content] if isinstance(user_content, str) else user_content
                        )
                        # Extract tool_use_result from UserMessage for metadata conversion
                        for user_block in user_blocks:
                            if isinstance(user_block, ToolResultBlock):
                                tc_id = user_block.tool_use_id
                                result_content = user_block.content
                                # Flush response parts
                                if current_response_parts:
                                    model_response = ModelResponse(parts=current_response_parts)
                                    model_messages.append(model_response)
                                    current_response_parts = []

                                # Get tool name from pending calls
                                tool_use = pending_tool_calls.pop(tc_id, None)
                                tool_name = _strip_mcp_prefix(
                                    tool_use.name if tool_use else "unknown"
                                )
                                # Create ToolReturnPart for the result
                                return_part = ToolReturnPart(
                                    tool_name=tool_name,
                                    content=result_content,
                                    tool_call_id=tc_id,
                                )
                                # Emit FunctionToolResultEvent (for session.py to complete UI)
                                yield FunctionToolResultEvent(result=return_part)
                                # Build metadata: prefer existing tool_metadata,
                                # then convert SDK result
                                tool_input = tool_use.input if tool_use else {}
                                metadata = tool_metadata.get(tc_id)
                                if not metadata and message.tool_use_result is not None:
                                    # Convert Claude Code SDK's tool_use_result to OpenCode format
                                    metadata = convert_to_opencode_metadata(
                                        tool_name, message.tool_use_result, tool_input
                                    )

                                # Also emit ToolCallCompleteEvent for consumers that expect it
                                yield ToolCallCompleteEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tc_id,
                                    tool_input=tool_input,
                                    tool_result=result_content,
                                    agent_name=self.name,
                                    message_id="",
                                    metadata=metadata,
                                )
                                # Add tool return as ModelRequest
                                model_messages.append(ModelRequest(parts=[return_part]))

                    # Handle StreamEvent for real-time streaming
                    elif isinstance(message, StreamEvent):
                        event_data = message.event
                        event_type = event_data.get("type")
                        index = event_data.get("index", 0)
                        content_block = event_data.get("content_block", {})
                        block_type = content_block.get("type")
                        delta = event_data.get("delta", {})
                        match event_type, block_type or delta.get("type"):
                            # content_block_start events
                            case "content_block_start", "text":
                                yield PartStartEvent.text(index=index, content="")

                            case "content_block_start", "thinking":
                                yield PartStartEvent.thinking(index=index, content="")

                            case "content_block_start", "tool_use":
                                # Emit ToolCallStartEvent early (args still streaming)
                                tc_id = content_block.get("id", "")
                                raw_tool_name = content_block.get("name", "")
                                tool_name = _strip_mcp_prefix(raw_tool_name)
                                tool_accumulator.start(tc_id, tool_name)
                                # Track for permission matching - callback uses raw name
                                self._pending_tool_call_ids[raw_tool_name] = tc_id
                                # Derive rich info with empty args for now
                                rich_info = derive_rich_tool_info(raw_tool_name, {})
                                emitted_tool_starts.add(tc_id)
                                yield ToolCallStartEvent(
                                    tool_call_id=tc_id,
                                    tool_name=tool_name,
                                    title=rich_info.title,
                                    kind=rich_info.kind,
                                    locations=[],  # No locations yet, args not complete
                                    content=rich_info.content,
                                    raw_input={},  # Empty, will be filled when complete
                                )

                            # content_block_delta events
                            case "content_block_delta", "text_delta":
                                if delta := delta.get("text", ""):
                                    text_delta = TextPartDelta(content_delta=delta)
                                    yield PartDeltaEvent(index=index, delta=text_delta)

                            case "content_block_delta", "thinking_delta":
                                if delta := delta.get("thinking", ""):
                                    thinking_delta = ThinkingPartDelta(content_delta=delta)
                                    yield PartDeltaEvent(index=index, delta=thinking_delta)

                            case "content_block_delta", "input_json_delta":
                                # Accumulate tool argument JSON fragments
                                if partial_json := delta.get("partial_json", ""):
                                    # Find which tool call this belongs to by index
                                    for tc_id in tool_accumulator._calls:
                                        tool_accumulator.add_args(tc_id, partial_json)
                                        tool_delta = ToolCallPartDelta(
                                            args_delta=partial_json,
                                            tool_call_id=tc_id,
                                        )
                                        yield PartDeltaEvent(index=index, delta=tool_delta)
                                        break  # Only one tool call streams at a time

                            # content_block_stop events
                            case "content_block_stop", _:
                                # Emit with empty part - content was accumulated via deltas
                                yield PartEndEvent(index=index, part=TextPart(content=""))

                            case _:
                                pass  # Ignore other event types (message_start, etc.)

                        # Skip further processing for StreamEvent - don't duplicate
                        continue

                    # Convert to events and yield
                    # (skip AssistantMessage - already streamed via StreamEvent)
                    if not isinstance(message, AssistantMessage):
                        for event in claude_message_to_events(message, agent_name=self.name):
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

        except asyncio.CancelledError:
            self.log.info("Stream cancelled via CancelledError")
            # Emit partial response on cancellation
            # Build metadata with file tracking and SDK session ID
            metadata = file_tracker.get_metadata()
            if self._sdk_session_id:
                metadata["sdk_session_id"] = self._sdk_session_id
            content = "".join(i.content for i in current_response_parts if isinstance(i, TextPart))
            response_msg = ChatMessage[TResult](
                content=content,  # type: ignore[arg-type]
                role="assistant",
                name=self.name,
                message_id=message_id or str(uuid.uuid4()),
                session_id=self.session_id,
                parent_id=user_msg.message_id,
                model_name=resolved_model or self.model_name,
                messages=model_messages,
                finish_reason="stop",
                metadata=metadata,
            )
            yield StreamCompleteEvent(message=response_msg)
            # Post-processing handled by base class
            return

        except Exception as e:
            yield RunErrorEvent(message=str(e), run_id=run_id, agent_name=self.name)
            raise

        finally:
            # Disconnect fork client if we created one
            if fork_client:
                try:
                    await fork_client.disconnect()
                except Exception as e:  # noqa: BLE001
                    self.log.warning("Error disconnecting fork client", error=e)

        # Flush any remaining response parts
        if current_response_parts:
            model_messages.append(ModelResponse(parts=current_response_parts))

        # Determine final content - use structured output if available
        content = "".join(i.content for i in current_response_parts if isinstance(i, TextPart))
        final_content: TResult = (
            result_message.structured_output  # type: ignore[assignment]
            if self._output_type is not str and result_message and result_message.structured_output
            else content
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
            session_id=self.session_id,
            parent_id=user_msg.message_id,
            model_name=resolved_model or self.model_name,
            messages=model_messages,
            cost_info=cost_info,
            usage=request_usage or RequestUsage(),
            response_time=result_message.duration_ms / 1000 if result_message else None,
            finish_reason="stop" if self._cancelled else None,
            metadata=metadata,
        )

        # Emit stream complete - post-processing handled by base class
        yield StreamCompleteEvent[TResult](message=chat_message)

    async def _interrupt(self) -> None:
        """Call Claude SDK's native interrupt() to stop the query."""
        if self._client:
            try:
                await self._client.interrupt()
                self.log.info("Claude Code client interrupted")
            except Exception:
                self.log.exception("Failed to interrupt Claude Code client")

    async def set_model(self, model: AnthropicMaxModelName | str) -> None:
        """Set the model for future requests."""
        await self._set_mode(model, "model")

    async def set_permission_mode(self, mode: PermissionMode) -> None:
        """Set permission mode.

        Args:
            mode: Permission mode - "default", "acceptEdits", "plan", or "bypassPermissions"
        """
        self._permission_mode = mode
        # Update permission mode on client if connected
        if self._client:
            await self._client.set_permission_mode(mode)

    async def get_available_models(self) -> list[ModelInfo]:
        """Get available models for Claude Code agent (defined as static list)."""
        from agentpool.agents.claude_code_agent.static_info import MODELS

        return MODELS

    async def get_server_info(self) -> ClaudeCodeServerInfo:
        """Get server initialization info (models, commands, account info, ...) from Claude Code."""
        from agentpool.agents.claude_code_agent.models import ClaudeCodeServerInfo

        await self.ensure_initialized()
        assert self._client, "Client not connected after ensure_initialized"
        raw_info = await self._client.get_server_info()
        assert raw_info, "No server info returned (streaming mode should always provide it)"
        return ClaudeCodeServerInfo.model_validate(raw_info)

    async def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for Claude Code agent.

        Claude Code exposes permission modes and model selection.

        Returns:
            List of ModeCategory for permissions and models
        """
        from agentpool.agents.claude_code_agent.static_info import MODES, THINKING_MODES
        from agentpool.agents.modes import ModeCategory

        categories = [
            ModeCategory(
                id="mode",
                name="Mode",
                available_modes=MODES,
                current_mode_id=self._permission_mode or "default",
                category="mode",
            )
        ]
        # Model selection
        models = await self.get_available_models()
        categories.append(models_to_category(models, current_mode=self.model_name))
        # Thinking level selection
        categories.append(
            ModeCategory(
                id="thought_level",
                name="Thinking Level",
                available_modes=THINKING_MODES,
                current_mode_id=self._thinking_mode,
                category="thought_level",
            )
        )

        return categories

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """Handle permissions, model, and thinking_level mode switching."""
        from agentpool.agents.claude_code_agent.static_info import VALID_MODES
        from agentpool.agents.modes import ConfigOptionChanged

        if category_id == "mode":
            # Map mode_id to PermissionMode
            if mode_id not in VALID_MODES:
                raise UnknownModeError(mode_id, list(VALID_MODES))
            permission_mode: PermissionMode = mode_id  # type: ignore[assignment]
            self._permission_mode = permission_mode
            if self._client:  # Update SDK client if initialized
                await self.ensure_initialized()
                await self._client.set_permission_mode(permission_mode)
        elif category_id == "model":
            # Validate model exists
            if models := await self.get_available_models():
                valid_ids = {m.id_override if m.id_override else m.id for m in models}
                if mode_id not in valid_ids:
                    raise UnknownModeError(mode_id, list(valid_ids))
            # Set the model directly
            self._model = mode_id
            if self._client:
                await self.ensure_initialized()
                await self._client.set_model(mode_id)
        elif category_id == "thought_level":
            # Validate thinking mode
            if mode_id not in THINKING_MODE_TOKENS:
                raise UnknownModeError(mode_id, list(THINKING_MODE_TOKENS.keys()))
            self._thinking_mode = mode_id  # type: ignore[assignment]
            # Set thinking tokens via SDK
            if self._client:
                await self.ensure_initialized()
                tokens = THINKING_MODE_TOKENS[self._thinking_mode]
                await self._client.set_max_thinking_tokens(tokens)
        else:
            raise UnknownCategoryError(category_id)
        change = ConfigOptionChanged(config_id=category_id, value_id=mode_id)
        await self.state_updated.emit(change)
        self.log.info("Config option changed", config_id=category_id, value=mode_id)

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[SessionData]:
        """List sessions from Claude storage (~/.claude/projects/).

        Uses fast metadata reading that only parses timestamps and message counts,
        without loading full message content.
        """
        # Use fast metadata listing - avoids parsing all message content
        metadata_list = self._claude_storage.list_session_metadata(project_path=cwd)
        result: list[SessionData] = []
        default_cwd = str(self.env.cwd or Path.cwd())
        for meta in metadata_list:
            # Parse timestamps
            now = get_now()
            created_at = (
                parse_iso_timestamp(meta.first_timestamp, fallback=now)
                if meta.first_timestamp
                else now
            )
            last_active = (
                parse_iso_timestamp(meta.last_timestamp, fallback=created_at)
                if meta.last_timestamp
                else created_at
            )

            session_data = SessionData(
                session_id=meta.session_id,
                agent_name=self.name,
                cwd=meta.cwd or default_cwd,
                created_at=created_at,
                last_active=last_active,
                metadata={"title": meta.title, "message_count": meta.message_count}
                if meta.title
                else {"message_count": meta.message_count},
            )
            result.append(session_data)

        # Sort by last_active, most recent first
        result.sort(key=lambda s: s.updated_at or "", reverse=True)
        return result if limit is None else result[:limit]

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load and restore a session from Claude storage (requires reconnect)."""
        try:  # Load conversation messages from Claude storage
            messages = await self._claude_storage.get_session_messages(session_id=session_id)
        except Exception:
            self.log.exception("Failed to load Claude session", session_id=session_id)
            return None
        else:
            if not messages:
                self.log.warning("No messages found in session", session_id=session_id)
                return None
            # Restore to conversation history
            self.conversation.chat_messages.clear()
            self.conversation.chat_messages.extend(messages)
            self.log.info("Session loaded", session_id=session_id, message_count=len(messages))
            # Set the SDK session ID so reconnect can resume this session
            self._sdk_session_id = session_id
            # Reconnect to Claude SDK with the loaded session to properly resume
            try:
                await self.reconnect(resume_session=True)
                self.log.info("Reconnected with loaded session", session_id=session_id)
            except Exception:
                error_msg = "Failed to reconnect with loaded session, continuing with local history"
                self.log.exception(error_msg, session_id=session_id)
                # Don't fail the load - we still have the conversation history locally

            # Build SessionData from loaded messages
            last_active = messages[-1].timestamp if messages[-1].timestamp else get_now()
            cwd = str(self.env.cwd or Path.cwd())
            # Try to extract cwd from message metadata
            for msg in reversed(messages):
                if (val := msg.metadata.get("cwd")) and isinstance(val, str):
                    cwd = val
                    break

            return SessionData(
                session_id=session_id,
                agent_name=self.name,
                cwd=cwd,
                created_at=messages[0].timestamp if messages[0].timestamp else last_active,
                last_active=last_active,
            )


if __name__ == "__main__":
    import os
    import time

    os.environ["ANTHROPIC_API_KEY"] = ""

    async def main() -> None:
        """Demo: Basic call to Claude Code."""
        async with ClaudeCodeAgent(name="demo", event_handlers=["detailed"]) as agent:
            # print("Response (streaming): ", end="", flush=True)
            # async for _ in agent.run_stream("What files are in the current directory?"):
            #     pass
            await agent.ensure_initialized()
            print(now := time.time())
            sessions = await agent.list_sessions()
            print(time.time() - now)
            print(sessions)

    anyio.run(main)
