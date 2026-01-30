"""App, project, path, and VCS routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import anyenv
from fastapi import APIRouter, HTTPException

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import (
    App,
    AppTimeInfo,
    PathInfo,
    Project,
    ProjectTime,
    ProjectUpdatedEvent,
    ProjectUpdateRequest,
    VcsInfo,
)
from agentpool_storage.project_store import ProjectStore


if TYPE_CHECKING:
    from agentpool.sessions.models import ProjectData


router = APIRouter(tags=["app"])


@router.get("/app")
async def get_app(state: StateDep) -> App:
    """Get app information."""
    working_path = Path(state.working_dir)
    is_git = (working_path / ".git").is_dir()
    path_info = PathInfo(
        config="",
        cwd=state.working_dir,
        data="",
        root=state.working_dir,
        state="",
    )
    time_info = AppTimeInfo(initialized=state.start_time)
    return App(git=is_git, hostname="localhost", path=path_info, time=time_info)


def _project_data_to_response(data: ProjectData) -> Project:
    """Convert ProjectData to OpenCode Project response."""
    working_path = Path(data.worktree)
    vcs_dir: str | None = None
    if data.vcs == "git":
        vcs_dir = str(working_path / ".git")
    elif data.vcs == "hg":
        vcs_dir = str(working_path / ".hg")

    return Project(
        id=data.project_id,
        worktree=data.worktree,
        vcs_dir=vcs_dir,
        vcs=data.vcs,
        time=ProjectTime(created=int(data.created_at.timestamp() * 1000)),
    )


async def _get_current_project(state: StateDep) -> ProjectData:
    """Get or create the current project from storage."""
    storage = state.pool.storage
    project_store = ProjectStore(storage)
    return await project_store.get_or_create(state.working_dir)


@router.get("/project")
async def list_projects(state: StateDep) -> list[Project]:
    """List all projects."""
    storage = state.pool.storage
    project_store = ProjectStore(storage)
    projects = await project_store.list_recent(limit=50)
    return [_project_data_to_response(p) for p in projects]


@router.get("/project/current")
async def get_project_current(state: StateDep) -> Project:
    """Get current project."""
    project = await _get_current_project(state)
    return _project_data_to_response(project)


@router.patch("/project/{project_id}")
async def update_project(project_id: str, update: ProjectUpdateRequest, state: StateDep) -> Project:
    """Update project metadata (name, settings).

    Emits a project.updated event when successful.

    Args:
        project_id: Project identifier
        update: Fields to update (name and/or settings)
        state: Server state

    Returns:
        Updated project data

    Raises:
        HTTPException: If project not found
    """
    store = ProjectStore(state.pool.storage)
    project_data = None
    # Update name if provided
    if update.name is not None:
        project_data = await store.set_name(project_id, update.name)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
    # Update settings if provided
    if update.settings:
        project_data = await store.update_settings(project_id, **update.settings)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
    # If neither name nor settings provided, just fetch the project
    if not project_data:
        project_data = await store.get_by_id(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

    # Convert to OpenCode Project model
    project = _project_data_to_response(project_data)
    # Broadcast event
    await state.broadcast_event(ProjectUpdatedEvent.create(project))
    return project


@router.get("/path")
async def get_path(state: StateDep) -> PathInfo:
    """Get current path info."""
    return PathInfo(config="", cwd=state.working_dir, data="", root=state.working_dir, state="")


async def _run_git_command(args: list[str], cwd: str) -> str | None:
    """Run a git command asynchronously and return stdout, or None on error."""
    try:
        proc = await anyenv.create_process("git", *args, cwd=cwd, stdout="pipe", stderr="pipe")
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return stdout.decode().strip()
    except OSError:
        return None


@router.get("/vcs")
async def get_vcs(state: StateDep) -> VcsInfo:
    """Get VCS info.

    TODO: For remote/ACP support, these git commands should run through
    state.env.execute_command() instead of subprocess.run() so they
    execute on the client side where the repository lives.
    """
    git_dir = Path(state.working_dir) / ".git"
    if not git_dir.is_dir():
        return VcsInfo(branch=None, dirty=False, commit=None)

    branch, commit, status = await asyncio.gather(
        _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], state.working_dir),
        _run_git_command(["rev-parse", "HEAD"], state.working_dir),
        _run_git_command(["status", "--porcelain"], state.working_dir),
    )

    return VcsInfo(branch=branch, dirty=bool(status), commit=commit)
