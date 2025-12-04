"""ACP agent presets for well-known ACP servers.

This module provides pre-configured settings for popular ACP-compatible agents,
enabling easy integration with typed configuration and LSP autocomplete support.

The presets build CLI arguments for spawning ACP servers, providing a typed
interface over command-line options.

Example YAML configuration:
    ```yaml
    agents:
      coder:
        type: claude
        claude:
          append_system_prompt: "Always write tests first."
          permission_mode: acceptEdits
          allowed_tools:
            - Read
            - Write
            - Bash(git:*)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


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

    add_dir: list[str] | None = Field(
        default=None,
        description="Additional directories to allow tool access to.",
    )

    def build_args(self) -> list[str]:
        """Build CLI arguments from settings.

        Returns:
            List of command-line arguments for claude-code-acp
        """
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
        if self.add_dir:
            args.extend(["--add-dir", *self.add_dir])

        return args


@dataclass
class ACPAgentPreset:
    """Definition of a well-known ACP agent preset."""

    name: str
    """Identifier for this preset (e.g., 'claude', 'aider')."""

    command: str
    """Command to spawn the ACP server."""

    base_args: list[str] = field(default_factory=list)
    """Default command-line arguments (before settings)."""

    description: str = ""
    """Human-readable description of the agent."""

    settings_model: type[BaseModel] | None = None
    """Pydantic model for typed settings (enables LSP autocomplete)."""

    env: dict[str, str] = field(default_factory=dict)
    """Default environment variables."""

    def build_command(self, settings: BaseModel | None = None) -> tuple[str, list[str]]:
        """Build full command with arguments.

        Args:
            settings: Optional settings model instance

        Returns:
            Tuple of (command, args)
        """
        args = list(self.base_args)

        if settings and hasattr(settings, "build_args"):
            args.extend(settings.build_args())

        return self.command, args


# Registry of well-known ACP agents
ACP_PRESETS: dict[str, ACPAgentPreset] = {
    "claude": ACPAgentPreset(
        name="claude",
        command="claude-code-acp",
        description="Claude Code via ACP (Anthropic's coding agent)",
        settings_model=ClaudeACPSettings,
    ),
    # Future presets:
    # "aider": ACPAgentPreset(
    #     name="aider",
    #     command="aider",
    #     base_args=["--acp"],
    #     description="Aider - AI pair programming in your terminal",
    #     settings_model=AiderACPSettings,
    # ),
    # "llmling": ACPAgentPreset(
    #     name="llmling",
    #     command="uv",
    #     base_args=["run", "llmling-agent", "serve-acp"],
    #     description="LLMling agent via ACP",
    #     settings_model=LLMlingACPSettings,
    # ),
}


def get_preset(name: str) -> ACPAgentPreset | None:
    """Get an ACP agent preset by name.

    Args:
        name: Preset identifier (e.g., 'claude')

    Returns:
        ACPAgentPreset if found, None otherwise
    """
    return ACP_PRESETS.get(name)


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(ACP_PRESETS.keys())
