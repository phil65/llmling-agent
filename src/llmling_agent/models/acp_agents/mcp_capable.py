"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

import json
from typing import Any, Literal, assert_never, cast

from pydantic import BaseModel, ConfigDict, Field
from tokonomics.model_discovery import ProviderType  # noqa: TC002

from llmling_agent.models.acp_agents.base import BaseACPAgentConfig
from llmling_agent_config.output_types import StructuredResponseConfig  # noqa: TC001
from llmling_agent_config.toolsets import ToolsetConfig  # noqa: TC001


ClaudeCodeModelName = Literal["default", "sonnet", "opus", "haiku", "sonnet[1m]", "opusplan"]
ClaudeCodeToolName = Literal[
    "AskUserQuestion",
    "Bash",
    "BashOutput",
    "Edit",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillShell",
    "NotebookEdit",
    "Read",
    "Skill",
    "SlashCommand",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
]
ClaudeCodePermissionmode = Literal["default", "acceptEdits", "bypassPermissions", "dontAsk", "plan"]


class MCPCapableACPAgentConfig(BaseACPAgentConfig):
    """Base class for ACP agents that support MCP (Model Context Protocol) servers.

    Extends BaseACPAgentConfig with MCP-specific capabilities including toolsets
    that can be exposed via an internal MCP bridge.
    """

    toolsets: list[ToolsetConfig] = Field(
        default_factory=list,
        title="Toolsets",
        examples=[
            [
                {"type": "subagent"},
                {"type": "agent_management"},
            ],
        ],
    )
    """Toolsets to expose to this ACP agent via MCP bridge.

    These toolsets will be started as an in-process MCP server and made
    available to the external ACP agent. This allows ACP agents to use
    internal llmling-agent toolsets like subagent delegation.
    """

    def build_mcp_config_json(self) -> str | None:
        """Convert inherited mcp_servers to standard MCP config JSON format.

        This format is used by Claude Desktop, VS Code extensions, and other tools.

        Returns:
            JSON string for MCP config, or None if no servers configured.
        """
        from urllib.parse import urlparse

        from llmling_agent_config.mcp_server import (
            SSEMCPServerConfig,
            StdioMCPServerConfig,
            StreamableHTTPMCPServerConfig,
        )

        servers = self.get_mcp_servers()
        if not servers:
            return None

        mcp_servers: dict[str, dict[str, Any]] = {}
        for idx, server in enumerate(servers):
            # Determine server name: explicit > derived
            match server:
                case _ if server.name:
                    name = server.name
                case StdioMCPServerConfig(args=[*_, last]):
                    name = last.split("/")[-1].split("@")[0]
                case StdioMCPServerConfig(command=cmd):
                    name = cmd
                case SSEMCPServerConfig(url=url) | StreamableHTTPMCPServerConfig(url=url):
                    name = urlparse(str(url)).hostname or f"server_{idx}"
                case _ as unreachable:
                    assert_never(unreachable)  # ty: ignore

            config: dict[str, Any]
            match server:
                case StdioMCPServerConfig(command=command, args=args):
                    config = {"command": command, "args": args}
                    if server.env:
                        config["env"] = server.get_env_vars()
                case SSEMCPServerConfig(url=url):
                    config = {"url": str(url), "transport": "sse"}
                case StreamableHTTPMCPServerConfig(url=url):
                    config = {"url": str(url), "transport": "http"}
                case _ as unreachable:
                    assert_never(unreachable)  # ty: ignore
            mcp_servers[name] = config

        if not mcp_servers:
            return None

        return json.dumps({"mcpServers": mcp_servers})


class ClaudeACPAgentConfig(MCPCapableACPAgentConfig):
    """Configuration for Claude Code via ACP.

    Provides typed settings for the claude-code-acp server.

    Note:
        If ANTHROPIC_API_KEY is set in your environment, Claude Code will use it
        directly instead of the subscription. To force subscription usage, set
        `env: {"ANTHROPIC_API_KEY": ""}` in the config.

    Example:
        ```yaml
        agents:
          coder:
            type: claude
            cwd: /path/to/project
            model: sonnet
            permission_mode: acceptEdits
            env:
              ANTHROPIC_API_KEY: ""  # Use subscription instead of API key
            allowed_tools:
              - Read
              - Write
              - Bash(git:*)
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Claude ACP Agent Configuration"})

    type: Literal["claude"] = Field("claude", init=False)
    """Discriminator for Claude ACP agent."""

    system_prompt: str | None = Field(
        default=None,
        title="System Prompt",
        examples=["You are a helpful coding assistant.", "Follow best practices for Python."],
    )
    """Custom system prompt (replaces default Claude Code prompt)."""

    append_system_prompt: str | None = Field(
        default=None,
        title="Append System Prompt",
        examples=["Always write tests.", "Prefer functional programming."],
    )
    """Text to append to the default system prompt."""

    model: ClaudeCodeModelName | None = Field(
        default=None,
        title="Model",
        examples=["sonnet", "opus", "claude-sonnet-4-20250514"],
    )
    """Model override. Use alias ('sonnet', 'opus') or full name."""

    permission_mode: ClaudeCodePermissionmode | None = Field(
        default=None,
        title="Permission Mode",
        examples=["acceptEdits", "bypassPermissions", "plan"],
    )
    """Permission handling mode for tool execution."""

    allowed_tools: list[ClaudeCodeToolName | str] | None = Field(
        default=None,
        title="Allowed Tools",
        examples=[["Read", "Write", "Bash(git:*)"], ["Edit", "Glob"]],
    )
    """Whitelist of allowed tools (e.g., ['Read', 'Write', 'Bash(git:*)'])."""

    disallowed_tools: list[ClaudeCodeToolName | str] | None = Field(
        default=None,
        title="Disallowed Tools",
        examples=[["WebSearch", "WebFetch"], ["KillShell"]],
    )
    """Blacklist of disallowed tools."""

    strict_mcp_config: bool = Field(default=False, title="Strict MCP Config")
    """Only use MCP servers from mcp_config, ignoring all other configs."""

    add_dir: list[str] | None = Field(
        default=None,
        title="Additional Directories",
        examples=[["/tmp", "/var/log"], ["/home/user/data"]],
    )
    """Additional directories to allow tool access to."""

    tools: list[ClaudeCodeToolName | str] | None = Field(
        default=None,
        title="Tools",
        examples=[["Bash", "Edit", "Read"], []],
    )
    """Available tools from built-in set. Empty list disables all tools."""

    fallback_model: ClaudeCodeModelName | None = Field(
        default=None,
        title="Fallback Model",
        examples=["sonnet", "haiku"],
    )
    """Fallback model when default is overloaded."""

    dangerously_skip_permissions: bool = Field(
        default=False,
        title="Dangerously Skip Permissions",
    )
    """Bypass all permission checks. Only for sandboxed environments."""

    output_type: str | StructuredResponseConfig | None = Field(
        default=None,
        title="Output Type",
        examples=[
            "json_response",
            {"response_schema": {"type": "import", "import_path": "mymodule:MyModel"}},
        ],
    )
    """Structured output configuration. Generates --output-format and --json-schema."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "claude-code-acp"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = []

        if self.system_prompt:
            args.extend(["--system-prompt", self.system_prompt])
        if self.append_system_prompt:
            args.extend(["--append-system-prompt", self.append_system_prompt])
        if self.model:
            args.extend(["--model", self.model])
        if self.permission_mode:
            args.extend(["--permission-mode", self.permission_mode])
        if self.allowed_tools:
            args.extend(["--allowed-tools", *self.allowed_tools])
        if self.disallowed_tools:
            args.extend(["--disallowed-tools", *self.disallowed_tools])

        # Convert inherited mcp_servers to Claude's --mcp-config JSON format
        mcp_json = self.build_mcp_config_json()
        if mcp_json:
            args.extend(["--mcp-config", mcp_json])

        if self.strict_mcp_config:
            args.append("--strict-mcp-config")
        if self.add_dir:
            args.extend(["--add-dir", *self.add_dir])
        if self.tools is not None:
            if self.tools:
                args.extend(["--tools", ",".join(self.tools)])
            else:
                args.extend(["--tools", ""])
        if self.fallback_model:
            args.extend(["--fallback-model", self.fallback_model])
        if self.dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")
        if self.output_type:
            args.extend(["--output-format", "json"])
            schema = self._resolve_json_schema()
            if schema:
                args.extend(["--json-schema", schema])

        return args

    def _resolve_json_schema(self) -> str | None:
        """Resolve output_type to a JSON schema string."""
        if self.output_type is None:
            return None
        if isinstance(self.output_type, str):
            # Named reference - caller must resolve
            return None
        # StructuredResponseConfig - resolve schema via get_schema()
        model_cls = cast(type[BaseModel], self.output_type.response_schema.get_schema())
        return json.dumps(model_cls.model_json_schema())

    @property
    def model_providers(self) -> list[ProviderType]:
        """Claude Code uses Anthropic models."""
        return ["anthropic"]


class GeminiACPAgentConfig(MCPCapableACPAgentConfig):
    """Configuration for Gemini CLI via ACP.

    Provides typed settings for the gemini CLI with ACP support.

    Example:
        ```yaml
        acp_agents:
          coder:
            type: gemini
            cwd: /path/to/project
            model: gemini-2.5-pro
            approval_mode: auto_edit
            allowed_tools:
              - read_file
              - write_file
              - terminal
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Gemini ACP Agent Configuration"})

    type: Literal["gemini"] = Field("gemini", init=False)
    """Discriminator for Gemini ACP agent."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["gemini-2.5-pro", "gemini-2.5-flash"],
    )
    """Model override."""

    approval_mode: Literal["default", "auto_edit", "yolo"] | None = Field(
        default=None,
        title="Approval Mode",
        examples=["auto_edit", "yolo"],
    )
    """Approval mode for tool execution."""

    sandbox: bool = Field(default=False, title="Sandbox")
    """Run in sandbox mode."""

    yolo: bool = Field(default=False, title="YOLO")
    """Automatically accept all actions."""

    allowed_tools: list[str] | None = Field(
        default=None,
        title="Allowed Tools",
        examples=[["read_file", "write_file", "terminal"], ["search"]],
    )
    """Tools allowed to run without confirmation."""

    allowed_mcp_server_names: list[str] | None = Field(
        default=None,
        title="Allowed MCP Server Names",
        examples=[["filesystem", "github"], ["slack"]],
    )
    """Allowed MCP server names."""

    extensions: list[str] | None = Field(
        default=None,
        title="Extensions",
        examples=[["python", "typescript"], ["rust", "go"]],
    )
    """List of extensions to use. If not provided, all are used."""

    include_directories: list[str] | None = Field(
        default=None,
        title="Include Directories",
        examples=[["/path/to/lib", "/path/to/shared"], ["./vendor"]],
    )
    """Additional directories to include in the workspace."""

    output_format: Literal["text", "json", "stream-json"] | None = Field(
        default=None,
        title="Output Format",
        examples=["json", "stream-json"],
    )
    """Output format."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "gemini"

    @property
    def model_providers(self) -> list[ProviderType]:
        """Gemini CLI uses Google Gemini models."""
        return ["gemini"]

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = ["--experimental-acp"]

        if self.model:
            args.extend(["--model", self.model])
        if self.approval_mode:
            args.extend(["--approval-mode", self.approval_mode])
        if self.sandbox:
            args.append("--sandbox")
        if self.yolo:
            args.append("--yolo")
        if self.allowed_tools:
            args.extend(["--allowed-tools", *self.allowed_tools])
        if self.allowed_mcp_server_names:
            args.extend(["--allowed-mcp-server-names", *self.allowed_mcp_server_names])
        if self.extensions:
            args.extend(["--extensions", *self.extensions])
        if self.include_directories:
            args.extend(["--include-directories", *self.include_directories])
        if self.output_format:
            args.extend(["--output-format", self.output_format])

        return args


class FastAgentACPAgentConfig(MCPCapableACPAgentConfig):
    """Configuration for fast-agent via ACP.

    Robust LLM agent with comprehensive MCP support.

    Supports MCP server integration via:
    - Internal bridge: Use `toolsets` field to expose llmling-agent toolsets
    - External servers: Use `url` field to connect to external MCP servers
    - Skills: Use `skills_dir` to specify custom skills directory

    Example:
        ```yaml
        acp_agents:
          coder:
            type: fast-agent
            cwd: /path/to/project
            model: claude-3.5-sonnet-20241022
            toolsets:
              - type: subagent
              - type: agent_management
            skills_dir: ./my-skills
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "FastAgent ACP Agent Configuration"})

    type: Literal["fast-agent"] = Field("fast-agent", init=False)
    """Discriminator for fast-agent ACP agent."""

    model: str = Field(
        ...,
        title="Model",
        examples=[
            "anthropic.claude-3-7-sonnet-latest",
            "openai.o3-mini.high",
            "openrouter.google/gemini-2.5-pro-exp-03-25:free",
        ],
    )
    """Model to use."""

    shell_access: bool = Field(default=False, title="Shell Access")
    """Enable shell and file access (-x flag)."""

    skills_dir: str | None = Field(
        default=None,
        title="Skills Directory",
        examples=["./skills", "/path/to/custom-skills", "~/.fast-agent/skills"],
    )
    """Override the default skills directory for custom agent skills."""

    url: str | None = Field(
        default=None,
        title="URL",
        examples=["https://huggingface.co/mcp", "http://localhost:8080"],
    )
    """MCP server URL to connect to. Can also be used with internal toolsets bridge."""

    auth: str | None = Field(
        default=None,
        title="Auth",
        examples=["bearer-token-123", "api-key-xyz"],
    )
    """Authentication token for MCP server."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "fast-agent-acp"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = []

        if self.model:
            args.extend(["--model", self.model])
        if self.shell_access:
            args.append("-x")
        if self.skills_dir:
            args.extend(["--skills-dir", self.skills_dir])

        # Collect URLs from toolsets bridge + user-specified URL
        urls: list[str] = []
        if self.url:
            urls.append(self.url)

        # Extract URLs from MCP config JSON (from toolsets)
        mcp_json = self.build_mcp_config_json()
        if mcp_json:
            mcp_config = json.loads(mcp_json)
            urls.extend(
                server_config["url"]
                for server_config in mcp_config.get("mcpServers", {}).values()
                if "url" in server_config
            )

        if urls:
            args.extend(["--url", ",".join(urls)])

        if self.auth:
            args.extend(["--auth", self.auth])

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """fast-agent supports multiple providers."""
        return ["openai", "anthropic", "gemini", "openrouter"]


class AuggieACPAgentConfig(MCPCapableACPAgentConfig):
    """Configuration for Auggie (Augment Code) via ACP.

    AI agent that brings Augment Code's power to the terminal.

    Example:
        ```yaml
        acp_agents:
          auggie:
            type: auggie
            cwd: /path/to/project
            model: auggie-sonnet
            workspace_root: /path/to/workspace
            rules: [rules.md]
            shell: bash
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Auggie ACP Agent Configuration"})

    type: Literal["auggie"] = Field("auggie", init=False)
    """Discriminator for Auggie ACP agent."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["auggie-sonnet", "auggie-haiku"],
    )
    """Model to use."""

    workspace_root: str | None = Field(
        default=None,
        title="Workspace Root",
        examples=["/path/to/workspace", "/home/user/project"],
    )
    """Workspace root (auto-detects git root if absent)."""

    rules: list[str] | None = Field(
        default=None,
        title="Rules",
        examples=[["rules.md", "coding-standards.md"], ["./custom-rules.txt"]],
    )
    """Additional rules files."""

    augment_cache_dir: str | None = Field(
        default=None,
        title="Augment Cache Dir",
        examples=["~/.augment", "/tmp/augment-cache"],
    )
    """Cache directory (default: ~/.augment)."""

    retry_timeout: int | None = Field(
        default=None,
        title="Retry Timeout",
        examples=[30, 60],
    )
    """Timeout for rate-limit retries (seconds)."""

    allow_indexing: bool = Field(default=False, title="Allow Indexing")
    """Skip the indexing confirmation screen in interactive mode."""

    augment_token_file: str | None = Field(
        default=None,
        title="Augment Token File",
        examples=["~/.augment/token", "/etc/augment/auth.token"],
    )
    """Path to file containing authentication token."""

    github_api_token: str | None = Field(
        default=None,
        title="GitHub API Token",
        examples=["~/.github/token", "/secrets/github.token"],
    )
    """Path to file containing GitHub API token."""

    permission: list[str] | None = Field(
        default=None,
        title="Permission",
        examples=[["bash:allow", "edit:confirm"], ["read:allow", "write:deny"]],
    )
    """Tool permissions with 'tool-name:policy' format."""

    remove_tool: list[str] | None = Field(
        default=None,
        title="Remove Tool",
        examples=[["deprecated-tool", "legacy-search"], ["old-formatter"]],
    )
    """Remove specific tools by name."""

    shell: Literal["bash", "zsh", "fish", "powershell"] | None = Field(
        default=None,
        title="Shell",
        examples=["bash", "zsh"],
    )
    """Select shell."""

    startup_script: str | None = Field(
        default=None,
        title="Startup Script",
        examples=["export PATH=$PATH:/usr/local/bin", "source ~/.bashrc"],
    )
    """Inline startup script to run before each command."""

    startup_script_file: str | None = Field(
        default=None,
        title="Startup Script File",
        examples=["~/.augment_startup.sh", "/etc/augment/init.sh"],
    )
    """Load startup script from file."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "auggie"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args = ["--acp"]

        if self.model:
            args.extend(["--model", self.model])
        if self.workspace_root:
            args.extend(["--workspace-root", self.workspace_root])
        if self.rules:
            for rule_file in self.rules:
                args.extend(["--rules", rule_file])
        if self.augment_cache_dir:
            args.extend(["--augment-cache-dir", self.augment_cache_dir])
        if self.retry_timeout is not None:
            args.extend(["--retry-timeout", str(self.retry_timeout)])
        if self.allow_indexing:
            args.append("--allow-indexing")
        if self.augment_token_file:
            args.extend(["--augment-token-file", self.augment_token_file])
        if self.github_api_token:
            args.extend(["--github-api-token", self.github_api_token])

        # Convert inherited mcp_servers to Auggie's --mcp-config format
        mcp_json = self.build_mcp_config_json()
        if mcp_json:
            args.extend(["--mcp-config", mcp_json])

        if self.permission:
            for perm in self.permission:
                args.extend(["--permission", perm])
        if self.remove_tool:
            for tool in self.remove_tool:
                args.extend(["--remove-tool", tool])
        if self.shell:
            args.extend(["--shell", self.shell])
        if self.startup_script:
            args.extend(["--startup-script", self.startup_script])
        if self.startup_script_file:
            args.extend(["--startup-script-file", self.startup_script_file])

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """Auggie uses Augment Code's models."""
        return []


class KimiACPAgentConfig(MCPCapableACPAgentConfig):
    """Configuration for Kimi CLI via ACP.

    Command-line agent from Moonshot AI with ACP support.

    Example:
        ```yaml
        acp_agents:
          kimi:
            type: kimi
            cwd: /path/to/project
            model: kimi-v1
            work_dir: /path/to/work
            yolo: true
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Kimi ACP Agent Configuration"})

    type: Literal["kimi"] = Field("kimi", init=False)
    """Discriminator for Kimi CLI ACP agent."""

    verbose: bool = Field(default=False, title="Verbose")
    """Print verbose information."""

    debug: bool = Field(default=False, title="Debug")
    """Log debug information."""

    agent_file: str | None = Field(
        default=None,
        title="Agent File",
        examples=["./my-agent.yaml", "/etc/kimi/agent.json"],
    )
    """Custom agent specification file."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["kimi-v1", "kimi-v2"],
    )
    """LLM model to use."""

    work_dir: str | None = Field(
        default=None,
        title="Work Dir",
        examples=["/path/to/work", "/tmp/kimi-workspace"],
    )
    """Working directory for the agent."""

    yolo: bool = Field(default=False, title="YOLO")
    """Automatically approve all actions."""

    thinking: bool | None = Field(default=None, title="Thinking")
    """Enable thinking mode if supported."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "kimi"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args = ["--acp"]

        if self.verbose:
            args.append("--verbose")
        if self.debug:
            args.append("--debug")
        if self.agent_file:
            args.extend(["--agent-file", self.agent_file])
        if self.model:
            args.extend(["--model", self.model])
        if self.work_dir:
            args.extend(["--work-dir", self.work_dir])

        # Convert inherited mcp_servers to Kimi's --mcp-config format
        mcp_json = self.build_mcp_config_json()
        if mcp_json:
            args.extend(["--mcp-config", mcp_json])

        if self.yolo:
            args.append("--yolo")
        if self.thinking is not None and self.thinking:
            args.append("--thinking")

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """Kimi uses Moonshot AI's models."""
        return []


# Union of all ACP agent config types
MCPCapableACPAgentConfigTypes = (
    ClaudeACPAgentConfig
    | GeminiACPAgentConfig
    | FastAgentACPAgentConfig
    | AuggieACPAgentConfig
    | KimiACPAgentConfig
)
