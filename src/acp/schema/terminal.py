"""Terminal schema definitions."""

from __future__ import annotations

from acp.schema.base import AnnotatedObject


class TerminalExitStatus(AnnotatedObject):
    """Exit status of a terminal command."""

    exit_code: int | None = None
    """The process exit code (may be null if terminated by signal)."""

    signal: str | None = None
    """The signal that terminated the process (may be null if exited normally)."""
