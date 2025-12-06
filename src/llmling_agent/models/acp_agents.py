"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from tokonomics.model_discovery import ProviderType  # noqa: TC002

from llmling_agent_config.nodes import NodeConfig
from llmling_agent_config.output_types import StructuredResponseConfig  # noqa: TC001


if TYPE_CHECKING:
    from anyenv.code_execution import ExecutionEnvironment
    from anyenv.code_execution.configs import (
        ExecutionEnvironmentConfig as ExecutionEnvironmentConfigType,
    )


ClaudeCodeModelName = Literal["default", "sonnet", "opus", "haiku", "sonnet[1m]", "opusplan"]
ClaudeCodeToolName = Literal["Read", "Grep", "Glob", "Bash"]
ClaudeCodePermissionmode = Literal["default", "acceptEdits", "bypassPermissions", "dontAsk", "plan"]


class BaseACPAgentConfig(NodeConfig):
    """Base configuration for all ACP agents.

    Provides common fields and the interface for building commands.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:terminal-16",
            "x-doc-title": "ACP Agent Configuration",
        }
    )

    cwd: str | None = Field(
        default=None,
        examples=["/path/to/project", "."],
    )
    """Working directory for the session."""

    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables to set."""

    execution_environment: Annotated[
        Literal["local", "docker", "e2b", "beam", "daytona", "srt"] | dict[str, Any],
        Field(
            default="local",
            examples=[
                "local",
                "docker",
                {"type": "e2b", "template": "python-sandbox"},
                {"type": "docker", "image": "python:3.13-slim"},
            ],
        ),
    ] = "local"
    """Execution environment config for ACP client operations (filesystem, terminals)."""

    allow_file_operations: bool = Field(default=True)
    """Whether to allow file read/write operations."""

    allow_terminal: bool = Field(default=True)
    """Whether to allow terminal operations."""

    auto_grant_permissions: bool = Field(default=True)
    """Whether to automatically grant all permission requests."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        raise NotImplementedError

    def get_args(self) -> list[str]:
        """Get command arguments."""
        raise NotImplementedError

    def get_execution_environment(self) -> ExecutionEnvironment:
        """Create execution environment from config."""
        from anyenv.code_execution.configs import (
            ExecutionEnvironmentConfig,
            LocalExecutionEnvironmentConfig,
        )
        from pydantic import TypeAdapter

        config: ExecutionEnvironmentConfigType
        match self.execution_environment:
            case str() as env_type:
                # Simple string like "local", "docker"
                config = TypeAdapter(ExecutionEnvironmentConfig).validate_python({"type": env_type})
            case dict() as env_dict:
                # Full config dict
                config = TypeAdapter(ExecutionEnvironmentConfig).validate_python(env_dict)
            case _:
                config = LocalExecutionEnvironmentConfig()

        return config.get_provider()

    @property
    def model_providers(self) -> list[ProviderType]:
        """Return the model providers used by this ACP agent.

        Override in subclasses to specify which providers the agent uses.
        Used for intelligent model discovery and fallback configuration.
        """
        return []

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
            # Determine server name: explicit > derived > fallback
            name: str
            if server.name:
                name = server.name
            elif isinstance(server, StdioMCPServerConfig):
                # Extract from command/args, e.g. "uvx mcp-server-fetch" -> "mcp-server-fetch"
                if server.args:
                    name = server.args[-1].split("/")[-1].split("@")[0]
                else:
                    name = server.command
            elif isinstance(server, SSEMCPServerConfig | StreamableHTTPMCPServerConfig):
                # Extract from URL hostname
                parsed = urlparse(str(server.url))
                name = parsed.hostname or f"server_{idx}"
            else:
                name = f"server_{idx}"

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
                case _:
                    continue
            mcp_servers[name] = config

        if not mcp_servers:
            return None

        return json.dumps({"mcpServers": mcp_servers})


class ACPAgentConfig(BaseACPAgentConfig):
    """Configuration for a custom ACP agent with explicit command.

    Use this for ACP servers that don't have a preset, or when you need
    full control over the command and arguments.

    Example:
        ```yaml
        agents:
          custom_agent:
            type: acp
            command: my-acp-server
            args: ["--mode", "coding"]
            cwd: /path/to/project
        ```
    """

    type: Literal["acp"] = Field("acp", init=False)
    """Discriminator for custom ACP agent."""

    command: str = Field(..., examples=["claude-code-acp", "aider", "my-custom-acp"])
    """Command to spawn the ACP server."""

    args: list[str] = Field(default_factory=list)
    """Arguments to pass to the command."""

    providers: list[ProviderType] = Field(default_factory=list)
    """Model providers this agent can use."""

    @property
    def model_providers(self) -> list[ProviderType]:
        """Return configured providers for custom ACP agents."""
        return list(self.providers)

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return self.command

    def get_args(self) -> list[str]:
        """Get command arguments."""
        return self.args


class ClaudeACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Claude Code via ACP.

    Provides typed settings for the claude-code-acp server.

    Example:
        ```yaml
        agents:
          coder:
            type: claude
            cwd: /path/to/project
            model: sonnet
            permission_mode: acceptEdits
            allowed_tools:
              - Read
              - Write
              - Bash(git:*)
        ```
    """

    type: Literal["claude"] = Field("claude", init=False)
    """Discriminator for Claude ACP agent."""

    system_prompt: str | None = Field(default=None)
    """Custom system prompt (replaces default Claude Code prompt)."""

    append_system_prompt: str | None = Field(default=None)
    """Text to append to the default system prompt."""

    model: ClaudeCodeModelName | None = Field(
        default=None,
        examples=["sonnet", "opus", "claude-sonnet-4-20250514"],
    )
    """Model override. Use alias ('sonnet', 'opus') or full name."""

    permission_mode: ClaudeCodePermissionmode | None = Field(default=None)
    """Permission handling mode for tool execution."""

    allowed_tools: list[ClaudeCodeToolName | str] | None = Field(default=None)
    """Whitelist of allowed tools (e.g., ['Read', 'Write', 'Bash(git:*)'])."""

    disallowed_tools: list[ClaudeCodeToolName | str] | None = Field(default=None)
    """Blacklist of disallowed tools."""

    strict_mcp_config: bool = Field(default=False)
    """Only use MCP servers from mcp_config, ignoring all other configs."""

    add_dir: list[str] | None = Field(default=None)
    """Additional directories to allow tool access to."""

    tools: list[ClaudeCodeToolName | str] | None = Field(
        default=None,
        examples=[["Bash", "Edit", "Read"], []],
    )
    """Available tools from built-in set. Empty list disables all tools."""

    fallback_model: ClaudeCodeModelName | None = Field(default=None, examples=["sonnet", "haiku"])
    """Fallback model when default is overloaded."""

    dangerously_skip_permissions: bool = Field(default=False)
    """Bypass all permission checks. Only for sandboxed environments."""

    output_type: str | StructuredResponseConfig | None = Field(
        default=None,
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


class GeminiACPAgentConfig(BaseACPAgentConfig):
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
        ```
    """

    type: Literal["gemini"] = Field("gemini", init=False)
    """Discriminator for Gemini ACP agent."""

    model: str | None = Field(
        default=None,
        examples=["gemini-2.5-pro", "gemini-2.5-flash"],
    )
    """Model override."""

    approval_mode: Literal["default", "auto_edit", "yolo"] | None = Field(default=None)
    """Approval mode for tool execution."""

    sandbox: bool = Field(default=False)
    """Run in sandbox mode."""

    yolo: bool = Field(default=False)
    """Automatically accept all actions."""

    allowed_tools: list[str] | None = Field(default=None)
    """Tools allowed to run without confirmation."""

    allowed_mcp_server_names: list[str] | None = Field(default=None)
    """Allowed MCP server names."""

    extensions: list[str] | None = Field(default=None)
    """List of extensions to use. If not provided, all are used."""

    include_directories: list[str] | None = Field(default=None)
    """Additional directories to include in the workspace."""

    output_format: Literal["text", "json", "stream-json"] | None = Field(default=None)
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

    type: Literal["codex"] = Field("codex", init=False)
    """Discriminator for Codex ACP agent."""

    model: str | None = Field(default=None, examples=["o3", "o4-mini"])
    """Model override."""

    sandbox_permissions: list[str] | None = Field(
        default=None,
        examples=[["disk-full-read-access"]],
    )
    """Sandbox permissions."""

    shell_environment_policy_inherit: Literal["all", "none"] | None = Field(default=None)
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

    type: Literal["opencode"] = Field("opencode", init=False)
    """Discriminator for OpenCode ACP agent."""

    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] | None = Field(default=None)
    """Log level."""

    print_logs: bool = Field(default=False)
    """Print logs to stderr."""

    port: int | None = Field(default=None)
    """Port to listen on."""

    hostname: str | None = Field(default=None, examples=["127.0.0.1", "0.0.0.0"])
    """Hostname to listen on."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "opencode"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        args: list[str] = ["acp"]

        if self.cwd:
            args.extend(["--cwd", self.cwd])
        if self.log_level:
            args.extend(["--log-level", self.log_level])
        if self.print_logs:
            args.append("--print-logs")
        if self.port is not None:
            args.extend(["--port", str(self.port)])
        if self.hostname:
            args.extend(["--hostname", self.hostname])

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


class FastAgentACPAgentConfig(BaseACPAgentConfig):
    """Configuration for fast-agent via ACP.

    Robust LLM agent with comprehensive MCP support.

    Example:
        ```yaml
        acp_agents:
          coder:
            type: fast-agent
            cwd: /path/to/project
            model: sonnet
            shell_access: true
        ```
    """

    type: Literal["fast-agent"] = Field("fast-agent", init=False)
    """Discriminator for fast-agent ACP agent."""

    model: str = Field(
        ...,
        examples=[
            "anthropic.claude-3-7-sonnet-latest",
            "openai.o3-mini.high",
            "openrouter.google/gemini-2.5-pro-exp-03-25:free",
        ],
    )
    """Model to use."""

    shell_access: bool = Field(default=False)
    """Enable shell and file access (-x flag)."""

    url: str | None = Field(default=None, examples=["https://huggingface.co/mcp"])
    """MCP server URL to connect to."""

    auth: str | None = Field(default=None)
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
        if self.url:
            args.extend(["--url", self.url])
        if self.auth:
            args.extend(["--auth", self.auth])

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """fast-agent supports multiple providers."""
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


class AuggieACPAgentConfig(BaseACPAgentConfig):
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

    type: Literal["auggie"] = Field("auggie", init=False)
    """Discriminator for Auggie ACP agent."""

    model: str | None = Field(default=None)
    """Model to use."""

    workspace_root: str | None = Field(default=None)
    """Workspace root (auto-detects git root if absent)."""

    rules: list[str] | None = Field(default=None)
    """Additional rules files."""

    augment_cache_dir: str | None = Field(default=None)
    """Cache directory (default: ~/.augment)."""

    retry_timeout: int | None = Field(default=None)
    """Timeout for rate-limit retries (seconds)."""

    allow_indexing: bool = Field(default=False)
    """Skip the indexing confirmation screen in interactive mode."""

    augment_token_file: str | None = Field(default=None)
    """Path to file containing authentication token."""

    github_api_token: str | None = Field(default=None)
    """Path to file containing GitHub API token."""

    permission: list[str] | None = Field(default=None)
    """Tool permissions with 'tool-name:policy' format."""

    remove_tool: list[str] | None = Field(default=None)
    """Remove specific tools by name."""

    shell: Literal["bash", "zsh", "fish", "powershell"] | None = Field(default=None)
    """Select shell."""

    startup_script: str | None = Field(default=None)
    """Inline startup script to run before each command."""

    startup_script_file: str | None = Field(default=None)
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

    type: Literal["cagent"] = Field("cagent", init=False)
    """Discriminator for Docker cagent ACP agent."""

    agent_file: str | None = Field(default=None)
    """Agent configuration file or registry reference."""

    code_mode_tools: bool = Field(default=False)
    """Provide a single tool to call other tools via Javascript."""

    env_from_file: list[str] | None = Field(default=None)
    """Set environment variables from file."""

    models_gateway: str | None = Field(default=None)
    """Set the models gateway address."""

    working_dir: str | None = Field(default=None)
    """Set the working directory for the session."""

    debug: bool = Field(default=False)
    """Enable debug logging."""

    otel: bool = Field(default=False)
    """Enable OpenTelemetry tracing."""

    log_file: str | None = Field(default=None)
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


class KimiACPAgentConfig(BaseACPAgentConfig):
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

    type: Literal["kimi"] = Field("kimi", init=False)
    """Discriminator for Kimi CLI ACP agent."""

    verbose: bool = Field(default=False)
    """Print verbose information."""

    debug: bool = Field(default=False)
    """Log debug information."""

    agent_file: str | None = Field(default=None)
    """Custom agent specification file."""

    model: str | None = Field(default=None)
    """LLM model to use."""

    work_dir: str | None = Field(default=None)
    """Working directory for the agent."""

    yolo: bool = Field(default=False)
    """Automatically approve all actions."""

    thinking: bool | None = Field(default=None)
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

    type: Literal["stakpak"] = Field("stakpak", init=False)
    """Discriminator for Stakpak ACP agent."""

    workdir: str | None = Field(default=None)
    """Run the agent in a specific directory."""

    verbose: bool = Field(default=False)
    """Enable verbose output."""

    debug: bool = Field(default=False)
    """Enable debug output."""

    disable_secret_redaction: bool = Field(default=False)
    """Disable secret redaction (WARNING: prints secrets to console)."""

    privacy_mode: bool = Field(default=False)
    """Enable privacy mode to redact private data."""

    study_mode: bool = Field(default=False)
    """Enable study mode to use the agent as a study assistant."""

    index_big_project: bool = Field(default=False)
    """Allow indexing of large projects (more than 500 supported files)."""

    enable_slack_tools: bool = Field(default=False)
    """Enable Slack tools (experimental)."""

    disable_mcp_mtls: bool = Field(default=False)
    """Disable mTLS (WARNING: uses unencrypted HTTP communication)."""

    enable_subagents: bool = Field(default=False)
    """Enable subagents."""

    subagent_config: str | None = Field(default=None)
    """Subagent configuration file subagents.toml."""

    allowed_tools: list[str] | None = Field(default=None)
    """Allow only the specified tools in the agent's context."""

    system_prompt_file: str | None = Field(default=None)
    """Read system prompt from file."""

    profile: str | None = Field(default=None)
    """Configuration profile to use."""

    model: Literal["smart", "eco"] | None = Field(default=None)
    """Choose agent model on startup."""

    config: str | None = Field(default=None)
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

    type: Literal["vtcode"] = Field("vtcode", init=False)
    """Discriminator for VT Code ACP agent."""

    model: str | None = Field(default=None)
    """LLM Model ID."""

    provider: Literal["gemini", "openai", "anthropic", "deepseek", "openrouter", "xai"] | None = (
        Field(default=None)
    )
    """LLM Provider."""

    api_key_env: str | None = Field(default=None)
    """API key environment variable."""

    workspace: str | None = Field(default=None)
    """Workspace root directory for file operations."""

    enable_tree_sitter: bool = Field(default=False)
    """Enable tree-sitter code analysis."""

    performance_monitoring: bool = Field(default=False)
    """Enable performance monitoring."""

    research_preview: bool = Field(default=False)
    """Enable research-preview features."""

    security_level: Literal["strict", "moderate", "permissive"] | None = Field(default=None)
    """Security level for tool execution."""

    show_file_diffs: bool = Field(default=False)
    """Show diffs for file changes in chat interface."""

    max_concurrent_ops: int | None = Field(default=None)
    """Maximum concurrent async operations."""

    api_rate_limit: int | None = Field(default=None)
    """Maximum API requests per minute."""

    max_tool_calls: int | None = Field(default=None)
    """Maximum tool calls per session."""

    debug: bool = Field(default=False)
    """Enable debug output for troubleshooting."""

    verbose: bool = Field(default=False)
    """Enable verbose logging."""

    config: str | None = Field(default=None)
    """Configuration file path."""

    log_level: Literal["error", "warn", "info", "debug", "trace"] | None = Field(default=None)
    """Log level."""

    theme: str | None = Field(default=None)
    """Select UI theme for ANSI styling."""

    skip_confirmations: bool = Field(default=False)
    """Skip safety confirmations."""

    full_auto: bool = Field(default=False)
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
        if self.debug:
            args.append("--debug")
        if self.verbose:
            args.append("--verbose")
        if self.config:
            args.extend(["--config", self.config])
        if self.log_level:
            args.extend(["--log-level", self.log_level])
        if self.theme:
            args.extend(["--theme", self.theme])
        if self.skip_confirmations:
            args.append("--skip-confirmations")
        if self.full_auto:
            args.append("--full-auto")

        return args

    @property
    def model_providers(self) -> list[ProviderType]:
        """VT Code supports multiple providers."""
        return ["openai", "anthropic", "gemini"]


# Union of all ACP agent config types
ACPAgentConfigTypes = Annotated[
    ACPAgentConfig
    | ClaudeACPAgentConfig
    | GeminiACPAgentConfig
    | CodexACPAgentConfig
    | OpenCodeACPAgentConfig
    | GooseACPAgentConfig
    | OpenHandsACPAgentConfig
    | FastAgentACPAgentConfig
    | AmpACPAgentConfig
    | AuggieACPAgentConfig
    | CagentACPAgentConfig
    | KimiACPAgentConfig
    | StakpakACPAgentConfig
    | VTCodeACPAgentConfig,
    Field(discriminator="type"),
]
