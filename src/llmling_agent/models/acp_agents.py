"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, Field

from llmling_agent_config.nodes import NodeConfig
from llmling_agent_config.output_types import StructuredResponseConfig  # noqa: TC001


if TYPE_CHECKING:
    from anyenv.code_execution import ExecutionEnvironment
    from anyenv.code_execution.configs import (
        ExecutionEnvironmentConfig as ExecutionEnvironmentConfigType,
    )


class BaseACPAgentConfig(NodeConfig):
    """Base configuration for all ACP agents.

    Provides common fields and the interface for building commands.
    """

    cwd: str | None = Field(
        default=None,
        description="Working directory for the session.",
        examples=["/path/to/project", "."],
    )
    """Working directory for the session."""

    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set.",
    )
    """Environment variables to set."""

    execution_environment: Annotated[
        Literal["local", "docker", "e2b", "beam", "daytona", "srt"] | dict[str, Any],
        Field(
            default="local",
            description="Execution environment for file/terminal operations.",
            examples=[
                "local",
                "docker",
                {"type": "e2b", "template": "python-sandbox"},
                {"type": "docker", "image": "python:3.13-slim"},
            ],
        ),
    ] = "local"
    """Execution environment config for ACP client operations (filesystem, terminals)."""

    allow_file_operations: bool = Field(
        default=True,
        description="Whether to allow file read/write operations.",
    )
    """Whether to allow file read/write operations."""

    allow_terminal: bool = Field(
        default=True,
        description="Whether to allow terminal operations.",
    )
    """Whether to allow terminal operations."""

    auto_grant_permissions: bool = Field(
        default=True,
        description="Whether to automatically grant all permission requests.",
    )
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
                case StdioMCPServerConfig():
                    config = {
                        "command": server.command,
                        "args": server.args,
                    }
                    if server.env:
                        config["env"] = server.get_env_vars()
                case SSEMCPServerConfig():
                    config = {
                        "url": str(server.url),
                        "transport": "sse",
                    }
                case StreamableHTTPMCPServerConfig():
                    config = {
                        "url": str(server.url),
                        "transport": "http",
                    }
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

    command: str = Field(
        ...,
        description="Command to spawn the ACP server.",
        examples=["claude-code-acp", "aider", "my-custom-acp"],
    )
    """Command to spawn the ACP server."""

    args: list[str] = Field(
        default_factory=list,
        description="Arguments to pass to the command.",
    )
    """Arguments to pass to the command."""

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

    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt (replaces default Claude Code prompt).",
    )

    append_system_prompt: str | None = Field(
        default=None,
        description="Text to append to the default system prompt.",
    )

    model: str | None = Field(
        default=None,
        description="Model override. Use alias ('sonnet', 'opus') or full name.",
        examples=["sonnet", "opus", "claude-sonnet-4-20250514"],
    )

    permission_mode: (
        Literal["default", "acceptEdits", "bypassPermissions", "dontAsk", "plan"] | None
    ) = Field(
        default=None,
        description="Permission handling mode for tool execution.",
    )

    allowed_tools: list[str] | None = Field(
        default=None,
        description="Whitelist of allowed tools (e.g., ['Read', 'Write', 'Bash(git:*)']).",
    )

    disallowed_tools: list[str] | None = Field(
        default=None,
        description="Blacklist of disallowed tools.",
    )

    mcp_config: list[str] | None = Field(
        default=None,
        description="MCP server config files or JSON strings to load.",
    )

    strict_mcp_config: bool = Field(
        default=False,
        description="Only use MCP servers from mcp_config, ignoring all other configs.",
    )

    add_dir: list[str] | None = Field(
        default=None,
        description="Additional directories to allow tool access to.",
    )

    tools: list[str] | None = Field(
        default=None,
        description="Available tools from built-in set. Empty list disables all tools.",
        examples=[["Bash", "Edit", "Read"], []],
    )

    fallback_model: str | None = Field(
        default=None,
        description="Fallback model when default is overloaded.",
        examples=["sonnet", "haiku"],
    )

    dangerously_skip_permissions: bool = Field(
        default=False,
        description="Bypass all permission checks. Only for sandboxed environments.",
    )

    output_type: str | StructuredResponseConfig | None = Field(
        default=None,
        description="Structured output configuration. Generates --output-format and --json-schema.",
        examples=[
            "json_response",
            {"response_schema": {"type": "import", "import_path": "mymodule:MyModel"}},
        ],
    )

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

        # Also include explicit mcp_config files/strings
        if self.mcp_config:
            args.extend(["--mcp-config", *self.mcp_config])
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
        description="Model override.",
        examples=["gemini-2.5-pro", "gemini-2.5-flash"],
    )

    approval_mode: Literal["default", "auto_edit", "yolo"] | None = Field(
        default=None,
        description="Approval mode for tool execution.",
    )

    sandbox: bool = Field(
        default=False,
        description="Run in sandbox mode.",
    )

    yolo: bool = Field(
        default=False,
        description="Automatically accept all actions.",
    )

    allowed_tools: list[str] | None = Field(
        default=None,
        description="Tools allowed to run without confirmation.",
    )

    allowed_mcp_server_names: list[str] | None = Field(
        default=None,
        description="Allowed MCP server names.",
    )

    extensions: list[str] | None = Field(
        default=None,
        description="List of extensions to use. If not provided, all are used.",
    )

    include_directories: list[str] | None = Field(
        default=None,
        description="Additional directories to include in the workspace.",
    )

    output_format: Literal["text", "json", "stream-json"] | None = Field(
        default=None,
        description="Output format.",
    )

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "gemini"

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

    model: str | None = Field(
        default=None,
        description="Model override.",
        examples=["o3", "o4-mini"],
    )

    sandbox_permissions: list[str] | None = Field(
        default=None,
        description="Sandbox permissions.",
        examples=[["disk-full-read-access"]],
    )

    shell_environment_policy_inherit: Literal["all", "none"] | None = Field(
        default=None,
        description="Shell environment inheritance policy.",
    )

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "npx"

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

    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] | None = Field(
        default=None,
        description="Log level.",
    )

    print_logs: bool = Field(
        default=False,
        description="Print logs to stderr.",
    )

    port: int | None = Field(
        default=None,
        description="Port to listen on.",
    )

    hostname: str | None = Field(
        default=None,
        description="Hostname to listen on.",
        examples=["127.0.0.1", "0.0.0.0"],
    )

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

    model: str | None = Field(
        default=None,
        description="Model to use.",
        examples=["sonnet", "kimi", "gpt-5-mini.low"],
    )

    shell_access: bool = Field(
        default=False,
        description="Enable shell and file access (-x flag).",
    )

    url: str | None = Field(
        default=None,
        description="MCP server URL to connect to.",
        examples=["https://huggingface.co/mcp"],
    )

    auth: str | None = Field(
        default=None,
        description="Authentication token for MCP server.",
    )

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


# Union of all ACP agent config types
ACPAgentConfigTypes = Annotated[
    ACPAgentConfig
    | ClaudeACPAgentConfig
    | GeminiACPAgentConfig
    | CodexACPAgentConfig
    | OpenCodeACPAgentConfig
    | GooseACPAgentConfig
    | OpenHandsACPAgentConfig
    | FastAgentACPAgentConfig,
    Field(discriminator="type"),
]
