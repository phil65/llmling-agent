"""Slash commands."""

from .docs_commands import get_docs_commands
from .terminal_commands import get_terminal_commands
from .acp_commands import get_acp_commands


def get_commands():
    """Get all ACP-specific commands."""
    return [*get_acp_commands(), *get_docs_commands(), *get_terminal_commands()]


__all__ = ["get_acp_commands", "get_docs_commands", "get_terminal_commands"]
