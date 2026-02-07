"""App, project, and path related models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


_APP_NAME = "opencode"


def _get_xdg_dir(env_var: str, default_subdir: str) -> str:
    """Get an XDG base directory, falling back to the spec default."""
    base = os.environ.get(env_var)
    if base:
        return str(Path(base) / _APP_NAME)
    return str(Path.home() / default_subdir / _APP_NAME)


class HealthResponse(OpenCodeBaseModel):
    """Response for /global/health endpoint."""

    healthy: bool = True
    version: str


class PathInfo(OpenCodeBaseModel):
    """Path information for the OpenCode instance.

    Maps to the upstream /path endpoint which returns XDG paths
    and the current working directory / worktree.
    """

    home: str
    """User home directory."""
    state: str
    """XDG state directory for opencode (e.g. ~/.local/state/opencode)."""
    config: str
    """XDG config directory for opencode (e.g. ~/.config/opencode)."""
    worktree: str
    """Git worktree root."""
    directory: str
    """Working directory."""

    @classmethod
    def for_directory(cls, directory: str, worktree: str | None = None) -> Self:
        """Build PathInfo for the given working directory.

        Args:
            directory: The working directory.
            worktree: Git worktree root. Falls back to directory if not provided.
        """
        return cls(
            home=str(Path.home()),
            state=_get_xdg_dir("XDG_STATE_HOME", ".local/state"),
            config=_get_xdg_dir("XDG_CONFIG_HOME", ".config"),
            worktree=worktree or directory,
            directory=directory,
        )


class AppTimeInfo(OpenCodeBaseModel):
    """App time information."""

    initialized: float | None = None


class App(OpenCodeBaseModel):
    """App information response."""

    git: bool = False
    hostname: str = "localhost"
    path: PathInfo
    time: AppTimeInfo


class ProjectTime(OpenCodeBaseModel):
    """Project time information."""

    created: int
    initialized: int | None = None


class Project(OpenCodeBaseModel):
    """Project information."""

    id: str
    worktree: str
    vcs_dir: str | None = None
    vcs: str | None = None  # "git" or None
    time: ProjectTime


class VcsInfo(OpenCodeBaseModel):
    """VCS (git) information."""

    branch: str | None = None
    dirty: bool = False
    commit: str | None = None


class ProjectUpdateRequest(OpenCodeBaseModel):
    """Request to update project metadata."""

    name: str | None = None
    """Optional friendly name for the project."""

    settings: dict[str, Any] | None = None
    """Optional project-specific settings to update."""
