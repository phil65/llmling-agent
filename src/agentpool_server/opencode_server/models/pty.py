"""PTY (Pseudo-Terminal) models."""

from __future__ import annotations

from typing import Literal

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class PtyInfo(OpenCodeBaseModel):
    """PTY session information."""

    id: str
    title: str
    command: str
    args: list[str]
    cwd: str
    status: Literal["running", "exited"]
    pid: int


class PtyCreateRequest(OpenCodeBaseModel):
    """Request to create a PTY session."""

    command: str | None = None
    args: list[str] | None = None
    cwd: str | None = None
    title: str | None = None
    env: dict[str, str] | None = None


class PtySize(OpenCodeBaseModel):
    """Terminal size."""

    rows: int
    cols: int


class PtyUpdateRequest(OpenCodeBaseModel):
    """Request to update a PTY session."""

    title: str | None = None
    size: PtySize | None = None
