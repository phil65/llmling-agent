"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from agentpool.models.acp_agents.base import BaseACPAgentConfig


class FastAgentACPAgentConfig(BaseACPAgentConfig):
    """Configuration for fast-agent via ACP.

    Robust LLM agent with comprehensive MCP support.

    Supports MCP server integration via:
    - Internal bridge: Use `toolsets` field to expose agentpool toolsets
    - External servers: Use `url` field to connect to external MCP servers
    - Skills: Use `skills_dir` to specify custom skills directory

    Example:
        ```yaml
        agents:
          coder:
            type: acp
            provider: fast-agent
            model: claude-3.5-sonnet-20241022
            tools:
              - type: subagent
              - type: agent_management
            skills_dir: ./my-skills
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "FastAgent ACP Agent Configuration"})

    provider: Literal["fast-agent"] = Field("fast-agent", init=False)
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
        if self.url:
            args.extend(["--url", self.url])
        if self.auth:
            args.extend(["--auth", self.auth])
        return args


class AuggieACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Auggie (Augment Code) via ACP.

    AI agent that brings Augment Code's power to the terminal.

    Example:
        ```yaml
        agents:
          auggie:
            type: acp
            provider: auggie
            model: auggie-sonnet
            rules: [rules.md]
            shell: bash
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Auggie ACP Agent Configuration"})

    provider: Literal["auggie"] = Field("auggie", init=False)
    """Discriminator for Auggie ACP agent."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["auggie-sonnet", "auggie-haiku"],
    )
    """Model to use."""

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


class KimiACPAgentConfig(BaseACPAgentConfig):
    """Configuration for Kimi CLI via ACP.

    Command-line agent from Moonshot AI with ACP support.

    Example:
        ```yaml
        agents:
          kimi:
            type: acp
            provider: kimi
            model: kimi-v1
            work_dir: /path/to/work
            yolo: true
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Kimi ACP Agent Configuration"})

    provider: Literal["kimi"] = Field("kimi", init=False)
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

    auto_approve: bool = Field(default=False, title="Auto Approve")
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
        if self.auto_approve:
            args.append("--yolo")
        if self.thinking is not None and self.thinking:
            args.append("--thinking")

        return args


class AgentpoolACPAgentConfig(BaseACPAgentConfig):
    """Configuration for agentpool's own ACP server.

    This allows using agentpool serve-acp as an ACP agent, with MCP bridge support
    for tool metadata preservation.

    Example:
        ```yaml
        acp_agents:
          my_agentpool:
            type: agentpool
            config_path: path/to/agent_config.yml
            agent: agent_name  # Optional: specific agent to use
            mcp_servers:
              - type: stdio
                command: mcp-server-filesystem
                args: ["--root", "/workspace"]
        ```
    """

    model_config = ConfigDict(title="Agentpool ACP Agent")

    provider: Literal["agentpool"] = Field("agentpool", init=False)
    """Discriminator for agentpool ACP agent."""

    config_path: str | None = None
    """Path to agentpool configuration file (optional)."""

    agent: str | None = None
    """Specific agent name to use from config (defaults to first agent)."""

    load_skills: bool = True
    """Load client-side skills from .claude/skills directory."""

    def get_command(self) -> str:
        """Get the command to run agentpool serve-acp."""
        return "agentpool"

    def get_args(self) -> list[str]:
        """Build command arguments for agentpool serve-acp."""
        args = ["serve-acp"]

        if self.config_path:
            args.append(self.config_path)
        if self.agent:
            args.extend(["--agent", self.agent])
        if not self.load_skills:
            args.append("--no-skills")
        return args


# Union of all ACP agent config types
MCPCapableACPAgentConfigTypes = (
    FastAgentACPAgentConfig | AuggieACPAgentConfig | KimiACPAgentConfig | AgentpoolACPAgentConfig
)
