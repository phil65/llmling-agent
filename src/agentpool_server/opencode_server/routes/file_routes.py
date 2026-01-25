"""File operation routes."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, Any

import anyenv
from fastapi import APIRouter, HTTPException, Query

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import (
    FileContent,
    FileNode,
    FindMatch,
    Symbol,
)
from agentpool_server.opencode_server.models.file import SubmatchInfo


if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem


router = APIRouter(tags=["file"])


# Directories to skip when searching
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}

# Sensitive files that should never be exposed via the API
BLOCKED_FILES = {".env", ".env.local", ".env.production", ".env.development", ".env.test"}


def _validate_path(root: Path, user_path: str) -> Path:
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
    resolved_root = root.resolve()

    # Join and resolve the full path (this handles ../, symlinks, etc.)
    target = (root / user_path).resolve()

    # Check that the resolved path is within the resolved root
    try:
        target.relative_to(resolved_root)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied: path escapes project directory",
        ) from None

    # Check for blocked files
    if target.name in BLOCKED_FILES:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: {target.name} files are protected",
        )

    return target


def _validate_path_str(root: str, user_path: str) -> str:
    """Validate path for fsspec filesystem (string-based).

    Args:
        root: The root directory path as string.
        user_path: The user-provided relative path.

    Returns:
        The validated absolute path as string.

    Raises:
        HTTPException: If the path escapes root or is blocked.
    """
    validated = _validate_path(Path(root), user_path)
    return str(validated)


def _get_fs(state: StateDep) -> tuple[AsyncFileSystem, str]:
    """Get the fsspec filesystem from the agent's environment.

    Returns:
        Tuple of (filesystem, base_path).
        base_path is the root directory to use for operations.
    """
    fs = state.agent.env.get_fs()
    # Use env's cwd if set, otherwise use state.working_dir
    base_path = state.agent.env.cwd or state.working_dir
    return (fs, base_path)


def _is_local_fs(fs: AsyncFileSystem) -> bool:
    """Check if filesystem is a local filesystem."""
    return getattr(fs, "local_file", False)


def _has_ripgrep() -> bool:
    """Check if ripgrep is available."""
    return shutil.which("rg") is not None


async def _search_with_ripgrep(
    pattern: str,
    base_path: str,
    max_matches: int = 100,
) -> list[FindMatch]:
    """Search using ripgrep for better performance on local filesystems.

    Args:
        pattern: Regex pattern to search for.
        base_path: Directory to search in.
        max_matches: Maximum number of matches to return.

    Returns:
        List of FindMatch objects.
    """
    # Build ripgrep command with JSON output
    cmd = ["rg", "--json", "--max-count", str(max_matches), "--no-binary"]
    # Add exclude patterns for SKIP_DIRS
    for skip_dir in SKIP_DIRS:
        cmd.extend(["--glob", f"!{skip_dir}/"])
    cmd.extend(["-e", pattern, base_path])
    # Run ripgrep asynchronously
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    matches: list[FindMatch] = []
    base_path_prefix = base_path.rstrip("/") + "/"
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            data = anyenv.load_json(line, return_type=dict)
            if data.get("type") != "match":
                continue

            match_data = data.get("data", {})
            path = match_data.get("path", {}).get("text", "")
            line_number = match_data.get("line_number", 0)
            line_text = match_data.get("lines", {}).get("text", "").rstrip("\n")
            absolute_offset = match_data.get("absolute_offset", 0)
            # Convert to relative path
            rel_path = path[len(base_path_prefix) :] if path.startswith(base_path_prefix) else path
            # Extract submatches
            submatches = []
            for sm in match_data.get("submatches", []):
                match_text = sm.get("match", {}).get("text", "")
                start = sm.get("start", 0)
                end = sm.get("end", 0)
                submatches.append(SubmatchInfo.create(match_text, start, end))

            matches.append(
                FindMatch.create(
                    path=rel_path,
                    lines=line_text.strip(),
                    line_number=line_number,
                    absolute_offset=absolute_offset,
                    submatches=submatches,
                )
            )

            if len(matches) >= max_matches:
                break
        except anyenv.JsonLoadError:
            continue

    return matches


async def _find_files_with_ripgrep(query: str, base_path: str, max_results: int = 100) -> list[str]:
    """Find files using ripgrep --files for better performance.

    Args:
        query: Glob pattern to match file names.
        base_path: Directory to search in.
        max_results: Maximum number of results to return.

    Returns:
        List of relative file paths.
    """
    # Build ripgrep command to list files matching glob
    cmd = ["rg", "--files"]
    # Add exclude patterns for SKIP_DIRS
    for skip_dir in SKIP_DIRS:
        cmd.extend(["--glob", f"!{skip_dir}/"])
    # Add the file name pattern as a glob
    # rg --files --glob supports matching anywhere in the path
    # If query doesn't contain glob chars, wrap it with * for substring matching
    glob_chars = {"*", "?", "[", "]"}
    if not any(c in query for c in glob_chars):
        query = f"*{query}*"
    # Use **/ prefix to match the filename in any directory
    cmd.extend(["--glob", f"**/{query}"])
    cmd.append(base_path)
    # Run ripgrep asynchronously
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    results: list[str] = []
    base_path_prefix = base_path.rstrip("/") + "/"
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        # Convert to relative path
        rel_path = line[len(base_path_prefix) :] if line.startswith(base_path_prefix) else line
        results.append(rel_path)
        if len(results) >= max_results:
            break
    return sorted(results)


@router.get("/file")
async def list_files(state: StateDep, path: str = Query(default="")) -> list[FileNode]:
    """List files in a directory."""
    working_path = Path(state.working_dir)
    # Validate path if provided (empty path means root, which is always valid)
    target_p = _validate_path(working_path, path) if path else working_path.resolve()
    fs, _base_path = _get_fs(state)
    resolved_root = working_path.resolve()
    target = str(target_p)
    try:
        nodes = []
        for entry in await fs._ls(target, detail=True):
            full_name = entry.get("name", "")
            name = full_name.split("/")[-1]
            if not name:
                continue
            # Skip blocked files in directory listings
            if name in BLOCKED_FILES:
                continue
            node_type = "directory" if entry.get("type") == "directory" else "file"
            size = entry.get("size") if node_type == "file" else None
            # Build relative path from resolved root
            rel_path = os.path.relpath(full_name, resolved_root)
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
    fs, _base_path = _get_fs(state)
    full_path = str(target)
    try:
        content = await fs._cat_file(full_path)
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
    fs, base_path = _get_fs(state)
    # Fast path: use ripgrep for local filesystems
    if _is_local_fs(fs) and _has_ripgrep():
        return await _search_with_ripgrep(pattern, base_path, max_matches)
    # Slow path: manual file iteration via fsspec
    matches: list[FindMatch] = []
    regex = re.compile(pattern)
    try:
        # Use find to get all files recursively (limit depth to avoid scanning huge trees)
        for path in await fs._find(base_path, maxdepth=10, withdirs=False):
            if len(matches) >= max_matches:
                break
            # Skip directories we don't want to search
            if any(part in SKIP_DIRS for part in path.split("/")):
                continue
            # Get relative path
            rel_path = path[len(base_path) :].lstrip("/") if path.startswith(base_path) else path
            try:
                content = await fs._cat_file(path)
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
    fs, base_path = _get_fs(state)
    # Fast path: use ripgrep for local filesystems (files only, not dirs)
    if not include_dirs and _is_local_fs(fs) and _has_ripgrep():
        return await _find_files_with_ripgrep(query, base_path, max_results)
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
                # Get relative path
                rel_p = path[len(base_path) :].lstrip("/") if path.startswith(base_path) else path
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
