"""Configuration models for ACP (Agent Client Protocol) agents.

This module provides configuration classes for ACP agents:
- BaseACPAgentConfig: Common base with interface methods
- ACPAgentConfig: Custom ACP agent with explicit command/args
- ClaudeACPAgentConfig: Pre-configured Claude Code agent
- (Future: AiderACPAgentConfig, LLMlingACPAgentConfig, etc.)

All configs use discriminated unions via the `type` field.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llmling_agent_config.nodes import NodeConfig
from llmling_agent_config.output_types import StructuredResponseConfig


class ClaudeACPSettings(BaseModel):
    """Settings for claude-code-acp agent.

    These settings map to claude CLI arguments.
    See `claude --help` for full documentation.
    """

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

    def build_args(self) -> list[str]:
        """Build CLI arguments from settings."""
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
        model_cls = self.output_type.response_schema.get_schema()
        return json.dumps(model_cls.model_json_schema())


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

    Provides typed settings for the claude-code-acp server with
    LSP autocomplete support.

    Example:
        ```yaml
        agents:
          coder:
            type: claude
            cwd: /path/to/project
            claude:
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

    claude: ClaudeACPSettings = Field(
        default_factory=ClaudeACPSettings,
        description="Claude-specific settings.",
    )
    """Claude-specific settings (maps to CLI arguments)."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return "claude-code-acp"

    def get_args(self) -> list[str]:
        """Build command arguments from settings."""
        return self.claude.build_args()


# Union of all ACP agent config types
ACPAgentConfigTypes = Annotated[
    ACPAgentConfig | ClaudeACPAgentConfig,
    Field(discriminator="type"),
]
