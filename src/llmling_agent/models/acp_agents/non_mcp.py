"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field
from tokonomics.model_discovery import ProviderType  # noqa: TC002

from llmling_agent.models.acp_agents.base import BaseACPAgentConfig


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


class CodexACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Zed Codex via ACP.

    Provides typed settings for the codex-acp server.

    Example:
        ```yaml
        acp_agents:
          coder:
            type: codex
            cwd: /path/to/project
            model: o3
            sandbox_permissions:
              - disk-full-read-access
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Codex ACP Agent Configuration"})

    type: Literal["codex"] = Field("codex", init=False)
    """Discriminator for Codex ACP agent."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["o3", "o4-mini"],
    )
    """Model override."""

    sandbox_permissions: list[str] | None = Field(
        default=None,
        title="Sandbox Permissions",
        examples=[["disk-full-read-access"], ["network-access", "disk-write-access"]],
    )
    """Sandbox permissions."""

    shell_environment_policy_inherit: Literal["all", "none"] | None = Field(
        default=None,
        title="Shell Environment Policy Inherit",
        examples=["all", "none"],
    )
    """Shell environment inheritance policy."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "npx"

    @property
    def model_providers(self) -> list[ProviderType]:
        """Codex uses OpenAI models."""
        return ["openai"]

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = ["@zed-industries/codex-acp"]

        if self.model:
            args.extend(["-c", f'model="{self.model}"'])
        if self.sandbox_permissions:
            # Format as TOML array
            perms = ", ".join(f'"{p}"' for p in self.sandbox_permissions)
            args.extend(["-c", f"sandbox_permissions=[{perms}]"])
        if self.shell_environment_policy_inherit:
            args.extend([
                "-c",
                f"shell_environment_policy.inherit={self.shell_environment_policy_inherit}",
            ])

        return args


class OpenCodeACPAgentConfig(BaseACPAgentConfig):
    """Configuration for OpenCode via ACP.

    Provides typed settings for the opencode acp server.

    Example:
        ```yaml
        acp_agents:
          coder:
            type: opencode
            cwd: /path/to/project
            log_level: INFO
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "OpenCode ACP Agent Configuration"})

    type: Literal["opencode"] = Field("opencode", init=False)
    """Discriminator for OpenCode ACP agent."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "opencode"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = ["acp"]

        if self.cwd:
            args.extend(["--cwd", self.cwd])
        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """OpenCode supports multiple providers."""
        return ["openai", "anthropic", "gemini", "openrouter"]


class GooseACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Goose via ACP.

    Block's open-source coding agent.

    Example:
        ```yaml
        acp_agents:
          coder:
            type: goose
            cwd: /path/to/project
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Goose ACP Agent Configuration"})

    type: Literal["goose"] = Field("goose", init=False)
    """Discriminator for Goose ACP agent."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "goose"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        return ["acp"]

    @property
    def model_providers(self) -> list[ProviderType]:
        """Goose supports multiple providers."""
        return ["openai", "anthropic", "gemini", "openrouter"]


class MistralACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Mistral Agent via ACP.

    Example:
        ```yaml
        acp_agents:
          coder:
            type: mistral
            cwd: /path/to/project
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Mistral ACP Agent Configuration"})

    type: Literal["mistral"] = Field("mistral", init=False)
    """Discriminator for Mistral ACP agent."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "vibe-acp"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        return []

    @property
    def model_providers(self) -> list[ProviderType]:
        """Goose supports multiple providers."""
        return ["mistral"]


class OpenHandsACPAgentConfig(BaseACPAgentConfig):
    """Configuration for OpenHands via ACP.

    Open-source autonomous AI agent (formerly OpenDevin).

    Example:
        ```yaml
        acp_agents:
          coder:
            type: openhands
            cwd: /path/to/project
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "OpenHands ACP Agent Configuration"})

    type: Literal["openhands"] = Field("openhands", init=False)
    """Discriminator for OpenHands ACP agent."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "openhands"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        return ["acp"]

    @property
    def model_providers(self) -> list[ProviderType]:
        """OpenHands supports multiple providers."""
        return ["openai", "anthropic", "gemini", "openrouter"]


class AmpACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Amp (AmpCode) via ACP.

    ACP bridge adapter that spawns the Amp CLI internally. The amp-acp bridge
    itself has no CLI configuration options. It spawns `amp --no-notifications`
    and bridges the communication to ACP protocol.

    Configuration is done via environment variables:
    - AMP_EXECUTABLE: Path to amp binary (default: 'amp' from PATH)
    - AMP_PREFER_SYSTEM_PATH: Set to '1' to use system amp instead of npx version
    - AMP_API_KEY: API key for Amp service
    - AMP_URL: URL for Amp service (default: https://ampcode.com/)
    - AMP_SETTINGS_FILE: Path to settings file

    For amp CLI configuration (permissions, MCP servers, etc.), use the amp
    settings file at ~/.config/amp/settings.json

    Example:
        ```yaml
        acp_agents:
          amp:
            type: amp
            cwd: /path/to/project
            env:
              AMP_EXECUTABLE: /usr/local/bin/amp
              AMP_PREFER_SYSTEM_PATH: "1"
              AMP_API_KEY: your-api-key
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Amp ACP Agent Configuration"})

    type: Literal["amp"] = Field("amp", init=False)
    """Discriminator for Amp ACP agent."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP bridge server."""
        return "npx"

    def get_args(self) -> list[str]:
        """Build command arguments for amp-acp bridge."""
        return ["-y", "amp-acp"]

    @property
    def model_providers(self) -> list[ProviderType]:
        """Amp supports multiple providers."""
        return ["openai", "anthropic", "gemini"]


class CagentACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Docker cagent via ACP.

    Agent Builder and Runtime by Docker Engineering.

    Example:
        ```yaml
        acp_agents:
          cagent:
            type: cagent
            cwd: /path/to/project
            agent_file: ./agent.yaml
            code_mode_tools: true
            working_dir: /path/to/work
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Cagent ACP Agent Configuration"})

    type: Literal["cagent"] = Field("cagent", init=False)
    """Discriminator for Docker cagent ACP agent."""

    agent_file: str | None = Field(
        default=None,
        title="Agent File",
        examples=["./agent.yaml", "registry.docker.io/my-agent:latest"],
    )
    """Agent configuration file or registry reference."""

    code_mode_tools: bool = Field(default=False, title="Code Mode Tools")
    """Provide a single tool to call other tools via Javascript."""

    env_from_file: list[str] | None = Field(
        default=None,
        title="Env From File",
        examples=[[".env", ".env.production"], ["config/.env.local"]],
    )
    """Set environment variables from file."""

    models_gateway: str | None = Field(
        default=None,
        title="Models Gateway",
        examples=["http://localhost:8000", "https://api.example.com/models"],
    )
    """Set the models gateway address."""

    working_dir: str | None = Field(
        default=None,
        title="Working Dir",
        examples=["/path/to/project", "/home/user/workspace"],
    )
    """Set the working directory for the session."""

    debug: bool = Field(default=False, title="Debug")
    """Enable debug logging."""

    otel: bool = Field(default=False, title="OTEL")
    """Enable OpenTelemetry tracing."""

    log_file: str | None = Field(
        default=None,
        title="Log File",
        examples=["/var/log/cagent.log", "./debug.log"],
    )
    """Path to debug log file."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "cagent"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args = ["acp"]

        if self.agent_file:
            args.append(self.agent_file)
        if self.code_mode_tools:
            args.append("--code-mode-tools")
        if self.env_from_file:
            for env_file in self.env_from_file:
                args.extend(["--env-from-file", env_file])
        if self.models_gateway:
            args.extend(["--models-gateway", self.models_gateway])
        if self.working_dir:
            args.extend(["--working-dir", self.working_dir])
        if self.debug:
            args.append("--debug")
        if self.otel:
            args.append("--otel")
        if self.log_file:
            args.extend(["--log-file", self.log_file])

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """Cagent supports multiple providers via MCP."""
        return ["openai", "anthropic", "gemini"]


class StakpakACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Stakpak Agent via ACP.

    Terminal-native DevOps Agent in Rust with enterprise-grade security.

    Example:
        ```yaml
        acp_agents:
          stakpak:
            type: stakpak
            cwd: /path/to/project
            model: smart
            workdir: /path/to/work
            verbose: true
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Stakpak ACP Agent Configuration"})

    type: Literal["stakpak"] = Field("stakpak", init=False)
    """Discriminator for Stakpak ACP agent."""

    workdir: str | None = Field(
        default=None,
        title="Workdir",
        examples=["/path/to/workdir", "/home/user/projects"],
    )
    """Run the agent in a specific directory."""

    verbose: bool = Field(default=False, title="Verbose")
    """Enable verbose output."""

    debug: bool = Field(default=False, title="Debug")
    """Enable debug output."""

    disable_secret_redaction: bool = Field(default=False, title="Disable Secret Redaction")
    """Disable secret redaction (WARNING: prints secrets to console)."""

    privacy_mode: bool = Field(default=False, title="Privacy Mode")
    """Enable privacy mode to redact private data."""

    study_mode: bool = Field(default=False, title="Study Mode")
    """Enable study mode to use the agent as a study assistant."""

    index_big_project: bool = Field(default=False, title="Index Big Project")
    """Allow indexing of large projects (more than 500 supported files)."""

    enable_slack_tools: bool = Field(default=False, title="Enable Slack Tools")
    """Enable Slack tools (experimental)."""

    disable_mcp_mtls: bool = Field(default=False, title="Disable MCP mTLS")
    """Disable mTLS (WARNING: uses unencrypted HTTP communication)."""

    enable_subagents: bool = Field(default=False, title="Enable Subagents")
    """Enable subagents."""

    subagent_config: str | None = Field(
        default=None,
        title="Subagent Config",
        examples=["subagents.toml", "/etc/stakpak/subagents.toml"],
    )
    """Subagent configuration file subagents.toml."""

    allowed_tools: list[str] | None = Field(
        default=None,
        title="Allowed Tools",
        examples=[["bash", "edit", "read"], ["search", "browse"]],
    )
    """Allow only the specified tools in the agent's context."""

    system_prompt_file: str | None = Field(
        default=None,
        title="System Prompt File",
        examples=["prompt.txt", "/etc/stakpak/system.txt"],
    )
    """Read system prompt from file."""

    profile: str | None = Field(
        default=None,
        title="Profile",
        examples=["default", "production", "development"],
    )
    """Configuration profile to use."""

    model: Literal["smart", "eco"] | None = Field(
        default=None,
        title="Model",
        examples=["smart", "eco"],
    )
    """Choose agent model on startup."""

    config: str | None = Field(
        default=None,
        title="Config",
        examples=["config.toml", "/etc/stakpak/config.toml"],
    )
    """Custom path to config file."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "stakpak"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args = ["acp"]

        if self.workdir:
            args.extend(["--workdir", self.workdir])
        if self.verbose:
            args.append("--verbose")
        if self.debug:
            args.append("--debug")
        if self.disable_secret_redaction:
            args.append("--disable-secret-redaction")
        if self.privacy_mode:
            args.append("--privacy-mode")
        if self.study_mode:
            args.append("--study-mode")
        if self.index_big_project:
            args.append("--index-big-project")
        if self.enable_slack_tools:
            args.append("--enable-slack-tools")
        if self.disable_mcp_mtls:
            args.append("--disable-mcp-mtls")
        if self.enable_subagents:
            args.append("--enable-subagents")
        if self.subagent_config:
            args.extend(["--subagent-config", self.subagent_config])
        if self.allowed_tools:
            for tool in self.allowed_tools:
                args.extend(["--tool", tool])
        if self.system_prompt_file:
            args.extend(["--system-prompt-file", self.system_prompt_file])
        if self.profile:
            args.extend(["--profile", self.profile])
        if self.model:
            args.extend(["--model", self.model])
        if self.config:
            args.extend(["--config", self.config])

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """Stakpak supports multiple providers."""
        return ["openai", "anthropic", "gemini"]


class VTCodeACPAgentConfig(BaseACPAgentConfig):
    """Configuration for VT Code via ACP.

    Rust-based terminal coding agent with semantic code intelligence.

    Example:
        ```yaml
        acp_agents:
          vtcode:
            type: vtcode
            cwd: /path/to/project
            model: gemini-2.5-flash-preview-05-20
            provider: gemini
            workspace: /path/to/workspace
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "VTCode ACP Agent Configuration"})

    type: Literal["vtcode"] = Field("vtcode", init=False)
    """Discriminator for VT Code ACP agent."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["gemini-2.5-flash-preview-05-20", "gpt-4o", "claude-3-5-sonnet"],
    )
    """LLM Model ID."""

    provider: Literal["gemini", "openai", "anthropic", "deepseek", "openrouter", "xai"] | None = (
        Field(
            default=None,
            title="Provider",
            examples=["gemini", "openai"],
        )
    )
    """LLM Provider."""

    api_key_env: str | None = Field(
        default=None,
        title="API Key Env",
        examples=["GEMINI_API_KEY", "OPENAI_API_KEY"],
    )
    """API key environment variable."""

    workspace: str | None = Field(
        default=None,
        title="Workspace",
        examples=["/path/to/workspace", "/home/user/projects"],
    )
    """Workspace root directory for file operations."""

    enable_tree_sitter: bool = Field(default=False, title="Enable Tree-Sitter")
    """Enable tree-sitter code analysis."""

    performance_monitoring: bool = Field(default=False, title="Performance Monitoring")
    """Enable performance monitoring."""

    research_preview: bool = Field(default=False, title="Research Preview")
    """Enable research-preview features."""

    security_level: Literal["strict", "moderate", "permissive"] | None = Field(
        default=None,
        title="Security Level",
        examples=["strict", "moderate"],
    )
    """Security level for tool execution."""

    show_file_diffs: bool = Field(default=False, title="Show File Diffs")
    """Show diffs for file changes in chat interface."""

    max_concurrent_ops: int | None = Field(
        default=None,
        title="Max Concurrent Ops",
        examples=[5, 10],
    )
    """Maximum concurrent async operations."""

    api_rate_limit: int | None = Field(
        default=None,
        title="API Rate Limit",
        examples=[60, 100],
    )
    """Maximum API requests per minute."""

    max_tool_calls: int | None = Field(
        default=None,
        title="Max Tool Calls",
        examples=[100, 500],
    )
    """Maximum tool calls per session."""

    config: str | None = Field(
        default=None,
        title="Config",
        examples=["config.toml", "/etc/vtcode/config.toml"],
    )
    """Configuration file path."""

    skip_confirmations: bool = Field(default=False, title="Skip Confirmations")
    """Skip safety confirmations."""

    full_auto: bool = Field(default=False, title="Full Auto")
    """Enable full-auto mode (no interaction)."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "vtcode"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args = ["acp"]

        if self.model:
            args.extend(["--model", self.model])
        if self.provider:
            args.extend(["--provider", self.provider])
        if self.api_key_env:
            args.extend(["--api-key-env", self.api_key_env])
        if self.workspace:
            args.extend(["--workspace", self.workspace])
        if self.enable_tree_sitter:
            args.append("--enable-tree-sitter")
        if self.performance_monitoring:
            args.append("--performance-monitoring")
        if self.research_preview:
            args.append("--research-preview")
        if self.security_level:
            args.extend(["--security-level", self.security_level])
        if self.show_file_diffs:
            args.append("--show-file-diffs")
        if self.max_concurrent_ops is not None:
            args.extend(["--max-concurrent-ops", str(self.max_concurrent_ops)])
        if self.api_rate_limit is not None:
            args.extend(["--api-rate-limit", str(self.api_rate_limit)])
        if self.max_tool_calls is not None:
            args.extend(["--max-tool-calls", str(self.max_tool_calls)])
        if self.config:
            args.extend(["--config", self.config])
        if self.skip_confirmations:
            args.append("--skip-confirmations")
        if self.full_auto:
            args.append("--full-auto")

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """VT Code supports multiple providers."""
        return ["openai", "anthropic", "gemini"]


class CursorACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Cursor via ACP.

    Cursor CLI agent with filesystem and terminal capabilities.
    See https://github.com/blowmage/cursor-agent-acp-npm

    Example:
        ```yaml
        acp_agents:
          coder:
            type: cursor
            cwd: /path/to/project
            session_dir: ~/.cursor-sessions
            timeout: 30000
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Cursor ACP Agent Configuration"})

    type: Literal["cursor"] = Field("cursor", init=False)
    """Discriminator for Cursor ACP agent."""

    config: str | None = Field(
        default=None,
        title="Config",
        examples=["config.json", "/etc/cursor/config.json"],
    )
    """Path to configuration file."""

    log_level: Literal["error", "warn", "info", "debug"] | None = Field(
        default=None,
        title="Log Level",
        examples=["info", "debug"],
    )
    """Logging level."""

    log_file: str | None = Field(
        default=None,
        title="Log File",
        examples=["/var/log/cursor.log", "./cursor-debug.log"],
    )
    """Log file path (logs to stderr by default)."""

    session_dir: str | None = Field(
        default=None,
        title="Session Dir",
        examples=["~/.cursor-sessions", "/tmp/cursor-sessions"],
    )
    """Session storage directory (default: ~/.cursor-sessions)."""

    timeout: int | None = Field(
        default=None,
        title="Timeout",
        examples=[30000, 60000],
    )
    """Cursor-agent timeout in milliseconds (default: 30000)."""

    retries: int | None = Field(
        default=None,
        title="Retries",
        examples=[3, 5],
    )
    """Number of retries for cursor-agent commands (default: 3)."""

    max_sessions: int | None = Field(
        default=None,
        title="Max Sessions",
        examples=[100, 200],
    )
    """Maximum number of concurrent sessions (default: 100)."""

    session_timeout: int | None = Field(
        default=None,
        title="Session Timeout",
        examples=[3600000, 7200000],
    )
    """Session timeout in milliseconds (default: 3600000)."""

    no_filesystem: bool = Field(default=False, title="No Filesystem")
    """Disable filesystem tools."""

    no_terminal: bool = Field(default=False, title="No Terminal")
    """Disable terminal tools."""

    max_processes: int | None = Field(
        default=None,
        title="Max Processes",
        examples=[5, 10],
    )
    """Maximum number of terminal processes (default: 5)."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "cursor-agent-acp"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = []

        if self.config:
            args.extend(["--config", self.config])
        if self.log_level:
            args.extend(["--log-level", self.log_level])
        if self.log_file:
            args.extend(["--log-file", self.log_file])
        if self.session_dir:
            args.extend(["--session-dir", self.session_dir])
        if self.timeout is not None:
            args.extend(["--timeout", str(self.timeout)])
        if self.retries is not None:
            args.extend(["--retries", str(self.retries)])
        if self.max_sessions is not None:
            args.extend(["--max-sessions", str(self.max_sessions)])
        if self.session_timeout is not None:
            args.extend(["--session-timeout", str(self.session_timeout)])
        if self.no_filesystem:
            args.append("--no-filesystem")
        if self.no_terminal:
            args.append("--no-terminal")
        if self.max_processes is not None:
            args.extend(["--max-processes", str(self.max_processes)])
        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """Cursor supports multiple providers."""
        return ["openai", "anthropic", "gemini", "openrouter"]


# Union of all ACP agent config types
RegularACPAgentConfigTypes = (
    CodexACPAgentConfig
    | OpenCodeACPAgentConfig
    | GooseACPAgentConfig
    | OpenHandsACPAgentConfig
    | AmpACPAgentConfig
    | CagentACPAgentConfig
    | StakpakACPAgentConfig
    | MistralACPAgentConfig
    | VTCodeACPAgentConfig
    | CursorACPAgentConfig
)
