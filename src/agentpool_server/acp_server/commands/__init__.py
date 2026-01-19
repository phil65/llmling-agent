"""Slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .debug_commands import get_debug_commands
from .docs_commands import get_docs_commands


if TYPE_CHECKING:
    from slashed import SlashedCommand


def get_commands() -> list[type[SlashedCommand]]:
    """Get all ACP-specific commands."""
    return [*get_debug_commands(), *get_docs_commands()]


__all__ = ["get_debug_commands", "get_docs_commands"]
