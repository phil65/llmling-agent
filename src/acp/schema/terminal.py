"""Terminal schema definitions."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from acp.base import Schema


class TerminalExitStatus(Schema):
    """Exit status of a terminal command."""

    field_meta: Any | None = None
    """Extension point for implementations."""

    exit_code: int | None = Field(ge=0)
    """The process exit code (may be null if terminated by signal)."""

    signal: str | None = None
    """The signal that terminated the process (may be null if exited normally)."""
