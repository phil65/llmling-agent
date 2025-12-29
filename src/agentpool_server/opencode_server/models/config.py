"""Config models."""

from __future__ import annotations

from typing import Any

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


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
