"""App, project, and path related models."""

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class HealthResponse(OpenCodeBaseModel):
    """Response for /global/health endpoint."""

    healthy: bool = True
    version: str


class PathInfo(OpenCodeBaseModel):
    """Path information for app/project."""

    config: str = ""
    cwd: str
    data: str = ""
    root: str
    state: str = ""


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
