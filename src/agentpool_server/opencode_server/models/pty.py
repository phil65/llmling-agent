"""PTY (Pseudo-Terminal) models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


if TYPE_CHECKING:
    from exxec import PtyInfo as ExxecPtyInfo


class PtyInfo(OpenCodeBaseModel):
    """PTY session information."""

    id: str
    title: str
    command: str
    args: list[str]
    cwd: str
    status: Literal["running", "exited"]
    pid: int

    @classmethod
    def from_exxec(cls, info: ExxecPtyInfo, title: str | None = None) -> PtyInfo:
        """Convert exxec PtyInfo to OpenCode PtyInfo model.

        Args:
            info: PtyInfo from exxec
            title: Optional title override

        Returns:
            OpenCode PtyInfo model
        """
        return cls(
            id=info.id,
            title=title or f"Terminal {info.id[-4:]}",
            command=info.command,
            args=info.args,
            cwd=info.cwd or "",
            status=info.status,
            pid=info.pid,
        )


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
