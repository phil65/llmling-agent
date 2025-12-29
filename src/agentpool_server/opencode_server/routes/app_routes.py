"""App, project, path, and VCS routes."""

from __future__ import annotations

from pathlib import Path
import subprocess

from fastapi import APIRouter

from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (
    App,
    AppTimeInfo,
    PathInfo,
    Project,
    ProjectTime,
    VcsInfo,
)


router = APIRouter(tags=["app"])


@router.get("/app")
async def get_app(state: StateDep) -> App:
    """Get app information."""
    working_path = Path(state.working_dir)
    return App(
        git=(working_path / ".git").is_dir(),
        hostname="localhost",
        path=PathInfo(
            config="",
            cwd=state.working_dir,
            data="",
            root=state.working_dir,
            state="",
        ),
        time=AppTimeInfo(initialized=state.start_time),
    )


def _make_project(state: StateDep) -> Project:
    """Create a Project from current state."""
    working_path = Path(state.working_dir)
    git_dir = working_path / ".git"
    return Project(
        id="default",
        worktree=state.working_dir,
        vcs_dir=str(git_dir) if git_dir.is_dir() else None,
        vcs="git" if git_dir.is_dir() else None,
        time=ProjectTime(created=int(state.start_time * 1000)),
    )


@router.get("/project")
async def list_projects(state: StateDep) -> list[Project]:
    """List all projects."""
    return [_make_project(state)]


@router.get("/project/current")
async def get_project_current(state: StateDep) -> Project:
    """Get current project."""
    return _make_project(state)


@router.get("/path")
async def get_path(state: StateDep) -> PathInfo:
    """Get current path info."""
    return PathInfo(
        config="",
        cwd=state.working_dir,
        data="",
        root=state.working_dir,
        state="",
    )


@router.get("/vcs")
async def get_vcs(state: StateDep) -> VcsInfo:
    """Get VCS info."""
    git_dir = Path(state.working_dir) / ".git"
    if not git_dir.is_dir():
        return VcsInfo(branch=None, dirty=False, commit=None)

    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=state.working_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=state.working_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=state.working_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return VcsInfo(branch=branch, dirty=dirty, commit=commit)
    except subprocess.CalledProcessError:
        return VcsInfo(branch=None, dirty=False, commit=None)
