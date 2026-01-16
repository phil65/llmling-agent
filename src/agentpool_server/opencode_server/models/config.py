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


class Config(OpenCodeBaseModel):
    """Server configuration.

    This is a simplified version - we only include fields the TUI needs.
    """

    # Model settings
    model: str | None = None
    small_model: str | None = None

    # Theme and UI
    theme: str | None = None
    username: str | None = None

    # Sharing
    share: str | None = None  # "manual", "auto", "disabled"

    # Provider configurations (simplified)
    provider: dict[str, Any] | None = None

    # MCP configurations
    mcp: dict[str, Any] | None = None

    # Instructions
    instructions: list[str] | None = None

    # Auto-update
    autoupdate: bool | None = None

    # Keybinds
    keybinds: Keybinds | None = None
