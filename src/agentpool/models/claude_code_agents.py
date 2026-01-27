"""Configuration models for Claude Code agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field
from schemez import Schema
from tokonomics.model_names import AnthropicMaxModelName  # noqa: TC002

from agentpool import log
from agentpool.models.fields import OutputTypeField, SystemPromptField  # noqa: TC001
from agentpool.resource_providers import StaticResourceProvider
from agentpool_config import (
    AnyToolConfig,  # noqa: TC001
    BaseToolConfig,
)
from agentpool_config.nodes import BaseAgentConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentpool.agents.claude_code_agent import ClaudeCodeAgent
    from agentpool.common_types import AnyEventHandlerType
    from agentpool.delegation import AgentPool
    from agentpool.resource_providers import ResourceProvider
    from agentpool.ui.base import InputProvider

logger = log.get_logger(__name__)

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
SettingSource = Literal["user", "project", "local"]
ToolName = Literal[
    "Task",
    "TaskOutput",
    "Bash",
    "Glob",
    "Grep",
    "ExitPlanMode",
    "Read",
    "Edit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "TodoWrite",
    "WebSearch",
    "KillShell",
    "AskUserQuestion",
    "Skill",
    "EnterPlanMode",
    "LSP",
    "Chrome",
]


class AgentDefinition(Schema):
    """Agent definition configuration."""

    description: str
    """Description of the agent."""

    prompt: str
    """Prompt for the agent."""

    tools: list[str] | None = None
    """List of tools the agent can use."""

    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None
    """Model to use for the agent."""


class ClaudeCodeAgentConfig(BaseAgentConfig):
    """Configuration for Claude Code agents.

    Claude Code agents use the Claude Agent SDK to interact with Claude Code CLI,
    enabling file operations, terminal access, and code editing capabilities.

    Example:
        ```yaml
        agent:
          coder:
            type: claude_code
            model: claude-sonnet-4-5
            allowed_tools:
              - Read
              - Write
              - Bash
            system_prompt: "You are a helpful coding assistant."
            max_turns: 10

          planner:
            permission_mode: plan
            max_thinking_tokens: 10000
            include_builtin_system_prompt: false
            system_prompt:
              - "You are a planning-only assistant."
              - "Focus on architecture decisions."
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "title": "Claude Code Agent Configuration",
            "x-icon": "simple-icons:anthropic",
        }
    )

    type: Literal["claude_code"] = Field(default="claude_code", init=False)
    """Top-level discriminator for agent type."""

    model: AnthropicMaxModelName | str | None = Field(
        default="opus",
        title="Model",
        examples=["claude-sonnet-4-5", "claude-opus-4", "claude-haiku-3-5"],
    )
    """Model to use for this agent. Defaults to Claude's default model."""

    allowed_tools: list[ToolName | str] | None = Field(
        default=None,
        title="Allowed Tools",
        examples=[["Read", "Write", "Bash"], ["Read", "Grep", "Glob"]],
    )
    """List of tool names the agent is allowed to use.

    If not specified, all tools are available (subject to permission_mode).
    Common tools: Read, Write, Edit, Bash, Glob, Grep, Task, WebFetch, etc.
    """

    disallowed_tools: list[ToolName | str] | None = Field(
        default=None,
        title="Disallowed Tools",
        examples=[["Bash", "Write"], ["Task"]],
    )
    """List of tool names the agent is NOT allowed to use.

    Takes precedence over allowed_tools if both are specified.
    """

    system_prompt: SystemPromptField = None
    """System prompt for the agent. Can be a string or list of strings/prompt configs.

    By default, this is appended to Claude Code's builtin system prompt.
    Set `include_builtin_system_prompt: false` to use only your custom prompt.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/system_prompts_configuration/
    """

    include_builtin_system_prompt: bool = Field(
        default=True,
        title="Include Builtin System Prompt",
    )
    """Whether to include Claude Code's builtin system prompt.

    - true (default): `system_prompt` is appended to the builtin
    - false: Only use `system_prompt`, discard the builtin
    """

    max_turns: int | None = Field(
        default=None,
        title="Max Turns",
        ge=1,
        examples=[5, 10, 20],
    )
    """Maximum number of conversation turns before stopping."""

    max_budget_usd: float | None = Field(
        default=None,
        title="Max Budget (USD)",
        ge=0.0,
        examples=[1.0, 5.0, 10.0],
    )
    """Maximum budget in USD before stopping.

    When set, the agent will stop once the estimated cost exceeds this limit.
    """

    max_thinking_tokens: int | None = Field(
        default=None,
        title="Max Thinking Tokens",
        ge=1000,
        examples=[5000, 10000, 50000],
    )
    """Maximum tokens for extended thinking mode.

    When set, enables Claude's extended thinking capability for more
    complex reasoning tasks.
    """

    permission_mode: PermissionMode | None = Field(
        default=None,
        title="Permission Mode",
        examples=["default", "acceptEdits", "plan", "bypassPermissions"],
    )
    """Permission handling mode:

    - "default": Ask for permission on each tool use
    - "acceptEdits": Auto-accept file edits but ask for other operations
    - "plan": Plan-only mode, no execution
    - "bypassPermissions": Skip all permission checks (use with caution)
    """

    output_type: OutputTypeField = None

    env_vars: dict[str, str] | None = Field(
        default=None,
        title="Environment Variables",
        examples=[{"ANTHROPIC_API_KEY": "", "DEBUG": "1"}],
    )
    """Environment variables to set for the agent process.

    Note: Set ANTHROPIC_API_KEY to empty string to force subscription usage.
    """

    add_dir: list[str] | None = Field(
        default=None,
        title="Additional Directories",
        examples=[["/tmp", "/var/log"], ["/home/user/data"]],
    )
    """Additional directories to allow tool access to."""

    builtin_subagents: dict[str, AgentDefinition] | None = Field(
        default=None,
        title="Built-in Subagents",
        examples=[{"sonnet": {"description": "Sonnet agent", "model": "sonnet"}}],
    )
    """Built-in subagents configuration."""

    builtin_tools: list[str] | None = Field(
        default=None,
        title="Built-in Tools",
        examples=[["Bash", "Edit", "Read"], ["Read", "Write", "LSP"], ["Bash", "Chrome"]],
    )
    """Available tools from Claude Code's built-in set.

    Empty list disables all tools. If not specified, all tools are available.
    Different from allowed_tools which filters an already-available set.

    Special tools:
    - "LSP": Enable Language Server Protocol support for code intelligence
      (go to definition, find references, symbol info, etc.)
    - "Chrome": Enable Claude in Chrome integration for browser control
      (opens, navigates, interacts with browser tabs)

    Both LSP and Chrome require additional setup in your environment.
    """

    fallback_model: AnthropicMaxModelName | str | None = Field(
        default=None,
        title="Fallback Model",
        examples=["claude-sonnet-4-5", "claude-haiku-3-5"],
    )
    """Fallback model when default is overloaded."""

    dangerously_skip_permissions: bool = Field(
        default=False,
        title="Dangerously Skip Permissions",
    )
    """Bypass all permission checks. Only for sandboxed environments."""

    setting_sources: list[SettingSource] | None = Field(
        default=None,
        title="Setting Sources",
        examples=[["user", "project"], ["local"], ["user", "project", "local"]],
    )
    """Setting sources to load configuration from.

    Controls which Claude Code settings files are loaded:
    - "user": User-level settings (~/.config/claude/settings.json)
    - "project": Project-level settings (.claude/settings.json in project root)
    - "local": Local settings (.claude/settings.local.json, git-ignored)

    If not specified, Claude Code will load all available settings.
    """

    use_subscription: bool = Field(default=False, title="Use Claude Subscription")
    """Force usage of Claude subscription instead of API key.

    When True, sets ANTHROPIC_API_KEY to empty string, forcing Claude Code
    to use your Claude.ai subscription for authentication instead of an API key.

    This is useful when:
    - You have a Claude Pro/Team subscription with higher rate limits
    - You want to use subscription credits instead of API credits
    - You're using features only available to subscribers

    Note: Requires an active Claude subscription and logged-in session.
    """

    tools: list[AnyToolConfig | str] = Field(
        default_factory=list,
        title="Tools",
        examples=[
            [
                {"type": "subagent"},
                {"type": "agent_management"},
                "webbrowser:open",
                {
                    "type": "import",
                    "import_path": "webbrowser:open",
                    "name": "web_browser",
                },
            ],
        ],
    )
    """Tools and toolsets to expose to this Claude Code agent via MCP bridge.

    Supports both single tools and toolsets. These will be started as an
    in-process MCP server and made available to Claude Code.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/tool_configuration/
    """

    def get_agent[TDeps](
        self,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        pool: AgentPool[Any] | None = None,
        deps_type: type[TDeps] | None = None,  # type: ignore[valid-type]
    ) -> ClaudeCodeAgent[TDeps, Any]:
        from agentpool.agents.claude_code_agent import ClaudeCodeAgent

        return ClaudeCodeAgent.from_config(
            self,
            event_handlers=event_handlers,
            input_provider=input_provider,
            agent_pool=pool,
            deps_type=deps_type,
        )

    def get_tool_providers(self) -> list[ResourceProvider]:
        """Get all resource providers for this agent's tools.

        Processes the unified tools list, separating:
        - Toolsets: Each becomes its own ResourceProvider
        - Single tools: Aggregated into a single StaticResourceProvider

        Returns:
            List of ResourceProvider instances
        """
        from agentpool.tools.base import Tool
        from agentpool_config.toolsets import BaseToolsetConfig

        providers: list[ResourceProvider] = []
        static_tools: list[Tool] = []

        for tool_config in self.tools:
            try:
                if isinstance(tool_config, BaseToolsetConfig):
                    providers.append(tool_config.get_provider())
                elif isinstance(tool_config, str):
                    static_tools.append(Tool.from_callable(tool_config))
                elif isinstance(tool_config, BaseToolConfig):
                    static_tools.append(tool_config.get_tool())
            except Exception:
                logger.exception("Failed to load tool", config=tool_config)
                continue

        if static_tools:
            providers.append(StaticResourceProvider(name="tools", tools=static_tools))

        return providers

    # Backward compatibility
    def get_toolset_providers(self) -> list[ResourceProvider]:
        """Deprecated: use get_tool_providers() instead."""
        return [
            p
            for p in self.get_tool_providers()
            if not isinstance(p, StaticResourceProvider) or p.name != "tools"
        ]

    def get_tool_provider(self) -> ResourceProvider | None:
        """Deprecated: use get_tool_providers() instead."""
        for p in self.get_tool_providers():
            if isinstance(p, StaticResourceProvider) and p.name == "tools":
                return p
        return None
