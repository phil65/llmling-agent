"""Slash commands."""

from __future__ import annotations

from .docs_commands import get_docs_commands
from .acp_commands import get_acp_commands
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slashed import SlashedCommand


def get_commands() -> list[type[SlashedCommand]]:
    """Get all ACP-specific commands."""
    return [*get_acp_commands(), *get_docs_commands()]


__all__ = ["get_acp_commands", "get_docs_commands"]
