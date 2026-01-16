"""Config models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class Keybinds(BaseModel):
    """Keybind configuration.

    Defines keyboard shortcuts for the TUI. Uses OpenCode's default keybinds.
    """

    leader: str = "ctrl+x"
    app_exit: str = "ctrl+c,ctrl+d,<leader>q"
    editor_open: str = "<leader>e"
    theme_list: str = "<leader>t"
    sidebar_toggle: str = "<leader>b"
    session_new: str = "<leader>n"
    session_list: str = "<leader>l"
    session_interrupt: str = "escape"
    session_compact: str = "<leader>c"
    command_list: str = "ctrl+k"
    model_list: str = "ctrl+m"
    agent_cycle: str = "ctrl+a"
    variant_cycle: str = "ctrl+t"
    prompt_clear: str = "ctrl+u"
    prompt_submit: str = "enter"
    prompt_paste: str = "ctrl+v"
    input_newline: str = "ctrl+j,shift+enter"


class WatcherConfig(OpenCodeBaseModel):
    """File watcher configuration."""

    ignore: list[str] | None = None
    """Glob patterns for files/directories to ignore.

    Default ignores (always applied): .git
    Example patterns: ["node_modules/**", "dist/**", "*.log"]
    """


class Config(OpenCodeBaseModel):
    """Server configuration.

    Maps to OpenCode's Config.Info schema. Fields are optional - TUI uses
    what it finds and falls back to defaults for missing fields.

    Reference: opencode/src/config/config.ts
    """

    # Model settings
    model: str | None = None
    """Model to use in format 'provider/model', e.g. 'anthropic/claude-sonnet-4'."""

    small_model: str | None = None
    """Small model for tasks like title generation, format 'provider/model'."""

    # Agent settings
    default_agent: str | None = None
    """Default agent to use when none specified. Must be a primary agent."""

    # Theme and UI
    theme: str | None = None
    """Theme name for the interface."""

    username: str | None = None
    """Custom username to display instead of system username."""

    # Sharing
    share: str | None = None
    """Sharing behavior: 'manual', 'auto', or 'disabled'."""

    # Provider configurations
    provider: dict[str, Any] | None = None
    """Custom provider configurations and model overrides."""

    disabled_providers: list[str] | None = None
    """Providers to disable from auto-loading."""

    enabled_providers: list[str] | None = None
    """When set, ONLY these providers are enabled. All others ignored."""

    # MCP configurations
    mcp: dict[str, Any] | None = None
    """MCP (Model Context Protocol) server configurations."""

    # Instructions
    instructions: list[str] | None = None
    """Custom instructions/system prompts."""

    # Auto-update
    autoupdate: bool | str | None = None
    """Auto-update: true, false, or 'notify' for notifications only."""

    # Keybinds
    keybinds: Keybinds | None = None
    """Custom keybind configurations."""

    # File watcher
    watcher: WatcherConfig | None = None
    """File watcher configuration for ignore patterns."""

    # Additional fields OpenCode supports (not typically needed for agentpool):
    # - $schema: JSON schema reference
    # - logLevel: Server-side log level
    # - tui: TUI-specific settings (scrollbar, sidebar width, etc.)
    # - server: Server config for 'opencode serve' command
    # - command: Custom slash command definitions
    # - plugin: Plugin paths to load
    # - snapshot: Enable/disable snapshots
    # - agent: Agent configurations (plan, build, explore, etc.)
    # - formatter: Code formatter settings
    # - lsp: LSP server configurations
