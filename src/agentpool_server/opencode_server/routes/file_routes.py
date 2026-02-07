"""File operation routes."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query
import ripgrep_rs

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import FileContent, FileNode, FindMatch, Symbol
from agentpool_server.opencode_server.models.file import SubmatchInfo


router = APIRouter(tags=["file"])


SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    "*.egg-info",
}
"""Directories to skip when searching."""

BLOCKED_FILES = {".env", ".env.local", ".env.production", ".env.development", ".env.test"}
"""Sensitive files that should never be exposed via the API."""


def _validate_path(root: Path | str, user_path: str) -> Path:
    """Validate and resolve a user-provided path, ensuring it stays within root.

    Args:
        root: The root directory (working_dir) that paths must stay within.
        user_path: The user-provided relative path.

    Returns:
        The resolved absolute path that is guaranteed to be within root.

    Raises:
        HTTPException: If the path escapes root or is blocked.
    """
    # Resolve the root to handle any symlinks in the root itself
    resolved_root = Path(root).resolve()
    # Join and resolve the full path (this handles ../, symlinks, etc.)
    target = (Path(root) / user_path).resolve()
    # Check that the resolved path is within the resolved root
    try:
        target.relative_to(resolved_root)
    except ValueError:
        detail = "Access denied: path escapes project directory"
        raise HTTPException(status_code=403, detail=detail) from None

    if target.name in BLOCKED_FILES:
        detail = f"Access denied: {target.name} files are protected"
        raise HTTPException(status_code=403, detail=detail)

    return target


def _skip_dir_globs() -> list[str]:
    """Build glob exclude patterns for SKIP_DIRS."""
    return [f"!{skip_dir}/" for skip_dir in SKIP_DIRS]


def _search_with_ripgrep(
    pattern: str,
    base_path: str,
    max_matches: int = 100,
) -> list[FindMatch]:
    """Search using ripgrep-rs for better performance on local filesystems.

    Args:
        pattern: Regex pattern to search for.
        base_path: Directory to search in.
        max_matches: Maximum number of matches to return.

    Returns:
        List of FindMatch objects.
    """
    results = ripgrep_rs.search_structured(
        [pattern],
        paths=[base_path],
        globs=_skip_dir_globs(),
        max_total=max_matches,
    )
    return [
        FindMatch.create(
            path=os.path.relpath(m.path, base_path),
            lines=m.line_text.strip(),
            line_number=m.line_number,
            absolute_offset=m.absolute_offset,
            submatches=[SubmatchInfo.create(sm.text, sm.start, sm.end) for sm in m.submatches],
        )
        for m in results
    ]


def _find_files_with_ripgrep(
    query: str,
    base_path: str,
    max_results: int = 100,
    *,
    include_dirs: bool = False,
) -> list[str]:
    """Find files using ripgrep-rs for better performance.

    Args:
        query: Glob pattern to match file names.
        base_path: Directory to search in.
        max_results: Maximum number of results to return.
        include_dirs: Whether to include directories in results.

    Returns:
        List of relative file paths.
    """
    # If query doesn't contain glob chars, wrap it with * for substring matching
    glob_chars = {"*", "?", "[", "]"}
    if not any(c in query for c in glob_chars):
        query = f"*{query}*"
    globs = [*_skip_dir_globs(), f"**/{query}"]
    results = ripgrep_rs.files(
        paths=[base_path],
        globs=globs,
        max_count=max_results,
        relative_to=base_path,
        include_dirs=include_dirs or None,
    )
    return sorted(results)


@router.get("/file")
async def list_files(state: StateDep, path: str = Query(default="")) -> list[FileNode]:
    """List files in a directory."""
    working_path = Path(state.working_dir)
    # Validate path if provided (empty path means root, which is always valid)
    target_p = _validate_path(working_path, path) if path else working_path.resolve()
    target = str(target_p)
    nodes = []
    try:
        for entry in await state.fs._ls(target, detail=True):
            full_name = entry.get("name", "")
            name = full_name.split("/")[-1]
            if not name or name in BLOCKED_FILES:
                continue
            # Skip directories that should be hidden (caches, build artifacts, etc.)
            if name in SKIP_DIRS or any(fnmatch.fnmatch(name, pat) for pat in SKIP_DIRS):
                continue
            node_type = "directory" if entry.get("type") == "directory" else "file"
            size = entry.get("size") if node_type == "file" else None
            # Build relative path from resolved root
            rel_path = os.path.relpath(full_name, working_path.resolve())
            nodes.append(FileNode(name=name, path=rel_path or name, type=node_type, size=size))
        return sorted(nodes, key=lambda n: (n.type != "directory", n.name.lower()))
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="Directory not found") from err


@router.get("/file/content")
async def read_file(state: StateDep, path: str = Query()) -> FileContent:
    """Read a file's content."""
    working_path = Path(state.working_dir)
    # Validate path - this checks for traversal, symlink escapes, and blocked files
    target = _validate_path(working_path, path)
    try:
        content = await state.fs._cat_file(str(target))
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return FileContent(path=path, content=content)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="File not found") from err
    except UnicodeDecodeError as err:
        raise HTTPException(status_code=400, detail="Cannot read binary file") from err


@router.get("/file/status")
async def get_file_status(state: StateDep) -> list[dict[str, Any]]:
    """Get status of tracked files.

    Returns empty list - file tracking not yet implemented.
    """
    _ = state
    return []


@router.get("/find")
async def find_text(state: StateDep, pattern: str = Query()) -> list[FindMatch]:
    """Search for text pattern in files using regex."""
    # Validate regex pattern
    try:
        re.compile(pattern)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Invalid regex: {e}") from e
    max_matches = 100
    base_path = state.base_path
    # Fast path: use ripgrep-rs library for local filesystems
    if state.is_local_fs:
        return _search_with_ripgrep(pattern, base_path, max_matches)
    # Slow path: manual file iteration via fsspec
    matches: list[FindMatch] = []
    regex = re.compile(pattern)
    try:
        # Use find to get all files recursively (limit depth to avoid scanning huge trees)
        for path in await state.fs._find(base_path, maxdepth=10, withdirs=False):
            if len(matches) >= max_matches:
                break
            # Skip directories we don't want to search
            if any(part in SKIP_DIRS for part in path.split("/")):
                continue
            # Get relative path using os.path.relpath for cross-platform support
            rel_path = os.path.relpath(path, base_path)
            try:
                content = await state.fs._cat_file(path)
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                for line_num, line in enumerate(content.splitlines(), 1):
                    for m in regex.finditer(line):
                        find_match = FindMatch.create(
                            path=rel_path,
                            lines=line.strip(),
                            line_number=line_num,
                            absolute_offset=m.start(),
                            submatches=[SubmatchInfo.create(m.group(), m.start(), m.end())],
                        )
                        matches.append(find_match)
                        if len(matches) >= max_matches:
                            break
                    if len(matches) >= max_matches:
                        break
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
    except Exception:  # noqa: BLE001
        pass

    return matches


@router.get("/find/file")
async def find_files(
    state: StateDep,
    query: str = Query(),
    dirs: str = Query(default="false"),
) -> list[str]:
    """Find files by name pattern (glob-style matching)."""
    include_dirs = dirs.lower() == "true"
    max_results = 100
    fs = state.fs
    base_path = state.base_path
    # Fast path: use ripgrep-rs library for local filesystems
    if state.is_local_fs:
        return _find_files_with_ripgrep(query, base_path, max_results, include_dirs=include_dirs)
    # Slow path: manual file iteration via fsspec
    results: list[str] = []
    try:
        # Get all entries recursively (limit depth to avoid scanning huge trees)
        for path in await fs._find(base_path, maxdepth=10, withdirs=include_dirs):
            if len(results) >= max_results:
                break
            # Skip directories we don't want to search
            parts = path.split("/")
            if any(part in SKIP_DIRS for part in parts):
                continue
            name = parts[-1] if parts else path
            if fnmatch.fnmatch(name, query):
                # Get relative path using os.path.relpath for cross-platform support
                rel_p = os.path.relpath(path, base_path)
                results.append(rel_p)
    except Exception:  # noqa: BLE001
        pass

    return sorted(results)


@router.get("/find/symbol")
async def find_symbols(state: StateDep, query: str = Query()) -> list[Symbol]:
    """Find workspace symbols.

    Returns empty list - LSP symbol search not yet implemented.
    """
    _ = state
    _ = query
    # TODO: Integrate with LSP or implement basic symbol extraction
    return []
