"""File watcher utilities using watchfiles."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Set as AbstractSet
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from watchfiles import Change, awatch


# Callback type for file change notifications
FileChangeCallback = Callable[[AbstractSet[tuple[Change, str]]], Awaitable[None]]


@dataclass
class FileWatcher:
    """Async file watcher using watchfiles.

    Watches specified paths for changes and calls a callback when changes occur.

    Example:
        ```python
        async def on_change(changes):
            for change_type, path in changes:
                print(f"{change_type}: {path}")

        watcher = FileWatcher(
            paths=["/path/to/.git/HEAD"],
            callback=on_change,
        )

        async with watcher:
            # Watcher is running
            await asyncio.sleep(60)
        # Watcher stopped
        ```
    """

    paths: list[str | Path]
    """Paths to watch (files or directories)."""

    callback: FileChangeCallback
    """Async callback invoked when changes are detected."""

    debounce: int = 100
    """Debounce time in milliseconds."""

    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    """Background watch task."""

    _stop_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    """Event to signal stop."""

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._task is not None:
            return  # Already running

        self._stop_event.clear()
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching for file changes."""
        if self._task is None:
            return

        self._stop_event.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _watch_loop(self) -> None:
        """Internal watch loop."""
        str_paths = [str(p) for p in self.paths]

        # Filter to only existing paths
        existing_paths = [p for p in str_paths if Path(p).exists()]
        if not existing_paths:
            return

        async for changes in awatch(
            *existing_paths,
            debounce=self.debounce,
            stop_event=self._stop_event,
        ):
            # Don't let callback errors kill the watcher
            with contextlib.suppress(Exception):
                await self.callback(changes)

    async def __aenter__(self) -> Self:
        """Start watcher on context enter."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop watcher on context exit."""
        await self.stop()


async def get_git_branch(repo_path: str | Path) -> str | None:
    """Get the current git branch name.

    Args:
        repo_path: Path to the git repository

    Returns:
        Branch name or None if not a git repo or on detached HEAD
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
    except OSError:
        return None
    else:
        branch = stdout.decode().strip()
        return branch if branch != "HEAD" else None


@dataclass
class GitBranchWatcher:
    """Watches for git branch changes.

    Monitors the .git/HEAD file and calls a callback when the branch changes.

    Example:
        ```python
        async def on_branch_change(branch: str | None):
            print(f"Branch changed to: {branch}")

        watcher = GitBranchWatcher(
            repo_path="/path/to/repo",
            callback=on_branch_change,
        )

        async with watcher:
            await asyncio.sleep(60)
        ```
    """

    repo_path: str | Path
    """Path to the git repository."""

    callback: Callable[[str | None], Awaitable[None]]
    """Async callback invoked with new branch name when branch changes."""

    _current_branch: str | None = field(default=None, repr=False)
    """Cached current branch."""

    _file_watcher: FileWatcher | None = field(default=None, repr=False)
    """Underlying file watcher."""

    async def start(self) -> None:
        """Start watching for branch changes."""
        repo = Path(self.repo_path)
        git_dir = repo / ".git"

        # Handle git worktrees - .git might be a file pointing to the real git dir
        if git_dir.is_file():
            content = git_dir.read_text().strip()
            if content.startswith("gitdir:"):
                git_dir = Path(content[7:].strip())

        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return

        # Get initial branch
        self._current_branch = await get_git_branch(self.repo_path)

        async def on_file_change(changes: AbstractSet[tuple[Change, str]]) -> None:
            new_branch = await get_git_branch(self.repo_path)
            if new_branch != self._current_branch:
                self._current_branch = new_branch
                await self.callback(new_branch)

        self._file_watcher = FileWatcher(
            paths=[head_file],
            callback=on_file_change,
        )
        await self._file_watcher.start()

    async def stop(self) -> None:
        """Stop watching for branch changes."""
        if self._file_watcher:
            await self._file_watcher.stop()
            self._file_watcher = None

    @property
    def current_branch(self) -> str | None:
        """Get the current cached branch name."""
        return self._current_branch

    async def __aenter__(self) -> Self:
        """Start watcher on context enter."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop watcher on context exit."""
        await self.stop()
