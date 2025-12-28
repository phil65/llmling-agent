"""File operation routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (
    FileContent,
    FileNode,
)


router = APIRouter(tags=["file"])


@router.get("/file")
async def list_files(
    state: StateDep,
    path: str = Query(default=""),
) -> list[FileNode]:
    """List files in a directory."""
    working_path = Path(state.working_dir)
    target = working_path / path if path else working_path

    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    nodes = []
    for entry in target.iterdir():
        node_type = "directory" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else None
        nodes.append(
            FileNode(
                name=entry.name,
                path=str(entry.relative_to(working_path)),
                type=node_type,
                size=size,
            )
        )

    return sorted(nodes, key=lambda n: (n.type != "directory", n.name.lower()))


@router.get("/file/content")
async def read_file(
    state: StateDep,
    path: str = Query(),
) -> FileContent:
    """Read a file's content."""
    target = Path(state.working_dir) / path

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        content = target.read_text(encoding="utf-8")
        return FileContent(path=path, content=content)
    except UnicodeDecodeError as err:
        raise HTTPException(status_code=400, detail="Cannot read binary file") from err
