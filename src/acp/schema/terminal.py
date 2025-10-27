"""Terminal schema definitions."""

from __future__ import annotations

from pydantic import Field

from acp.schema.base import AnnotatedObject


class TerminalExitStatus(AnnotatedObject):
    """Exit status of a terminal command."""

    exit_code: int | None = Field(ge=0)
    """The process exit code (may be null if terminated by signal)."""

    signal: str | None = None
    """The signal that terminated the process (may be null if exited normally)."""
