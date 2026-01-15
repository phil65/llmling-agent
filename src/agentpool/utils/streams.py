"""Stream utilities for merging async iterators."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

# Re-export FileTracker from new location for backwards compatibility
from agentpool.agents.events.processors import (
    FileTracker,
    FileTrackingProcessor,
    extract_file_path_from_tool_call,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


__all__ = [
    "FileChange",
    "FileOpsTracker",
    "FileTracker",
    "FileTrackingProcessor",
    "TodoEntry",
    "TodoPriority",
    "TodoStatus",
    "TodoTracker",
    "extract_file_path_from_tool_call",
    "merge_queue_into_iterator",
]


@asynccontextmanager
async def merge_queue_into_iterator[T, V](  # noqa: PLR0915
    primary_stream: AsyncIterator[T],
    secondary_queue: asyncio.Queue[V],
) -> AsyncIterator[AsyncIterator[T | V]]:
    """Merge a primary async stream with events from a secondary queue.

    Args:
        primary_stream: The main async iterator (e.g., provider events)
        secondary_queue: Queue containing secondary events (e.g., progress events)

    Yields:
        Async iterator that yields events from both sources in real-time.
        Secondary queue is fully drained before the iterator completes.

    Example:
        ```python
        progress_queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        async with merge_queue_into_iterator(provider_stream, progress_queue) as events:
            async for event in events:
                print(f"Got event: {event}")
        ```
    """
    # Create a queue for all merged events
    event_queue: asyncio.Queue[V | T | None] = asyncio.Queue()
    primary_done = asyncio.Event()
    primary_exception: BaseException | None = None
    # Track if we've signaled the end of streams
    end_signaled = False

    # Task to read from primary stream and put into merged queue
    async def primary_task() -> None:
        nonlocal primary_exception, end_signaled
        try:
            async for event in primary_stream:
                await event_queue.put(event)
        except asyncio.CancelledError:
            # Signal completion and unblock merged_events before re-raising
            primary_done.set()
            if not end_signaled:
                end_signaled = True
                await event_queue.put(None)
            raise
        except BaseException as e:  # noqa: BLE001
            primary_exception = e
        finally:
            primary_done.set()

    # Task to read from secondary queue and put into merged queue
    async def secondary_task() -> None:
        nonlocal end_signaled
        try:
            while not primary_done.is_set():
                try:
                    secondary_event = await asyncio.wait_for(secondary_queue.get(), timeout=0.01)
                    await event_queue.put(secondary_event)
                except TimeoutError:
                    continue
            # Drain any remaining events after primary completes
            while not secondary_queue.empty():
                try:
                    secondary_event = secondary_queue.get_nowait()
                    await event_queue.put(secondary_event)
                except asyncio.QueueEmpty:
                    break
            # Now signal end of all events (only if not already signaled)
            if not end_signaled:
                end_signaled = True
                await event_queue.put(None)
        except asyncio.CancelledError:
            # Still need to signal completion on cancel (only if not already signaled)
            if not end_signaled:
                end_signaled = True
                await event_queue.put(None)

    # Start both tasks
    primary_task_obj = asyncio.create_task(primary_task())
    secondary_task_obj = asyncio.create_task(secondary_task())

    try:
        # Create async iterator that drains the merged queue
        async def merged_events() -> AsyncIterator[V | T]:
            while True:
                event = await event_queue.get()
                if event is None:  # End of all streams
                    break
                yield event
            # Re-raise any exception from primary stream after draining
            if primary_exception is not None:
                raise primary_exception

        yield merged_events()

    finally:
        # Clean up tasks - cancel BOTH tasks
        primary_task_obj.cancel()
        secondary_task_obj.cancel()
        await asyncio.gather(primary_task_obj, secondary_task_obj, return_exceptions=True)


@dataclass
class FileChange:
    """Represents a single file change operation."""

    path: str
    """File path that was modified."""

    old_content: str | None
    """Content before change (None for new files)."""

    new_content: str | None
    """Content after change (None for deletions)."""

    operation: str
    """Type of operation: 'create', 'write', 'edit', 'delete'."""

    timestamp: float = field(default_factory=lambda: __import__("time").time())
    """Unix timestamp when the change occurred."""

    message_id: str | None = None
    """ID of the message that triggered this change (for revert-to-message)."""

    agent_name: str | None = None
    """Name of the agent that made this change."""

    def to_unified_diff(self) -> str:
        """Generate unified diff for this change.

        Returns:
            Unified diff string
        """
        import difflib

        old_lines = (self.old_content or "").splitlines(keepends=True)
        new_lines = (self.new_content or "").splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{self.path}",
            tofile=f"b/{self.path}",
        )
        return "".join(diff)


@dataclass
class FileOpsTracker:
    r"""Tracks file operations with full content for diff/revert support.

    Stores file changes with before/after content so they can be:
    - Displayed as diffs
    - Reverted to previous state
    - Filtered by message ID

    Example:
        ```python
        tracker = FileOpsTracker()

        # Record a file edit
        tracker.record_change(
            path="src/main.py",
            old_content="def foo(): pass",
            new_content="def foo():\\n    return 42",
            operation="edit",
        )

        # Get all diffs
        for change in tracker.changes:
            print(change.to_unified_diff())

        # Revert all changes
        for path, content in tracker.get_revert_operations():
            write_file(path, content)
        ```
    """

    changes: list[FileChange] = field(default_factory=list)
    """List of all recorded file changes in order."""

    reverted_changes: list[FileChange] = field(default_factory=list)
    """Changes that were reverted and can be restored with unrevert."""

    def record_change(
        self,
        path: str,
        old_content: str | None,
        new_content: str | None,
        operation: str,
        message_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Record a file change.

        Args:
            path: File path that was modified
            old_content: Content before change (None for new files)
            new_content: Content after change (None for deletions)
            operation: Type of operation ('create', 'write', 'edit', 'delete')
            message_id: Optional message ID that triggered this change
            agent_name: Optional name of the agent that made this change
        """
        self.changes.append(
            FileChange(
                path=path,
                old_content=old_content,
                new_content=new_content,
                operation=operation,
                message_id=message_id,
                agent_name=agent_name,
            )
        )

    def get_changes_for_path(self, path: str) -> list[FileChange]:
        """Get all changes for a specific file path.

        Args:
            path: File path to filter by

        Returns:
            List of changes for the given path
        """
        return [c for c in self.changes if c.path == path]

    def get_changes_since_message(self, message_id: str) -> list[FileChange]:
        """Get all changes since (and including) a specific message.

        Args:
            message_id: Message ID to start from

        Returns:
            List of changes from the given message onwards
        """
        result = []
        found = False
        for change in self.changes:
            if change.message_id == message_id:
                found = True
            if found:
                result.append(change)
        return result

    def get_modified_paths(self) -> set[str]:
        """Get set of all modified file paths.

        Returns:
            Set of file paths that have been modified
        """
        return {c.path for c in self.changes}

    def get_current_state(self) -> dict[str, str | None]:
        """Get the current state of all modified files.

        For each file, returns the content after all changes have been applied.
        Returns None for deleted files.

        Returns:
            Dict mapping path to current content (or None if deleted)
        """
        state: dict[str, str | None] = {}
        for change in self.changes:
            state[change.path] = change.new_content
        return state

    def get_original_state(self) -> dict[str, str | None]:
        """Get the original state of all modified files.

        For each file, returns the content before any changes were made.
        Returns None for files that were created (didn't exist).

        Returns:
            Dict mapping path to original content (or None if created)
        """
        state: dict[str, str | None] = {}
        for change in self.changes:
            if change.path not in state:
                state[change.path] = change.old_content
        return state

    def get_revert_operations(
        self, since_message_id: str | None = None
    ) -> list[tuple[str, str | None]]:
        """Get operations needed to revert changes.

        Returns list of (path, content) tuples in reverse order (newest first).
        If content is None, the file should be deleted.

        Args:
            since_message_id: If provided, only revert changes from this message onwards.
                              If None, revert all changes.

        Returns:
            List of (path, content_to_restore) tuples for revert
        """
        if since_message_id:
            changes = self.get_changes_since_message(since_message_id)
        else:
            changes = self.changes

        # Build map of path -> content to restore
        # For each path, we need the old_content of the FIRST change in our subset
        # (that's what the file looked like before any of these changes)
        original_for_path: dict[str, str | None] = {}
        for change in changes:
            if change.path not in original_for_path:
                original_for_path[change.path] = change.old_content

        return list(original_for_path.items())

    def get_combined_diff(self) -> str:
        """Get combined unified diff of all changes.

        Returns:
            Combined diff string for all file changes
        """
        diffs = []
        for change in self.changes:
            diff = change.to_unified_diff()
            if diff:
                diffs.append(diff)
        return "\n".join(diffs)

    def clear(self) -> None:
        """Clear all recorded changes."""
        self.changes.clear()

    def remove_changes_since_message(self, message_id: str) -> int:
        """Remove changes from a specific message onwards and store for unrevert.

        The removed changes are stored in `reverted_changes` so they can be
        restored later via `restore_reverted_changes()`.

        Args:
            message_id: Message ID to start removal from

        Returns:
            Number of changes removed
        """
        # Find the index of the first change with this message_id
        start_idx = None
        for i, change in enumerate(self.changes):
            if change.message_id == message_id:
                start_idx = i
                break

        if start_idx is None:
            return 0

        # Store removed changes for potential unrevert
        self.reverted_changes = self.changes[start_idx:]
        self.changes = self.changes[:start_idx]
        return len(self.reverted_changes)

    def get_unrevert_operations(self) -> list[tuple[str, str | None]]:
        """Get operations needed to restore reverted changes.

        Returns list of (path, content) tuples. The content is the new_content
        from each reverted change (what the file should contain after unrevert).

        Returns:
            List of (path, content_to_write) tuples for unrevert
        """
        if not self.reverted_changes:
            return []

        # For each path, we want the LAST new_content in the reverted changes
        # (that's what the file looked like before the revert)
        final_content: dict[str, str | None] = {}
        for change in self.reverted_changes:
            final_content[change.path] = change.new_content

        return list(final_content.items())

    def restore_reverted_changes(self) -> int:
        """Move reverted changes back to the main changes list.

        Returns:
            Number of changes restored
        """
        if not self.reverted_changes:
            return 0

        restored_count = len(self.reverted_changes)
        self.changes.extend(self.reverted_changes)
        self.reverted_changes = []
        return restored_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dict representation of all changes
        """
        return {
            "changes": [
                {
                    "path": c.path,
                    "operation": c.operation,
                    "timestamp": c.timestamp,
                    "message_id": c.message_id,
                    "agent_name": c.agent_name,
                    "has_old_content": c.old_content is not None,
                    "has_new_content": c.new_content is not None,
                }
                for c in self.changes
            ],
            "modified_paths": sorted(self.get_modified_paths()),
        }


# =============================================================================
# Todo/Plan Tracking
# =============================================================================

TodoPriority = Literal["high", "medium", "low"]
TodoStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class TodoEntry:
    """A single todo/plan entry.

    Represents a task that the agent intends to accomplish.
    """

    id: str
    """Unique identifier for this entry."""

    content: str
    """Human-readable description of what this task aims to accomplish."""

    status: TodoStatus = "pending"
    """Current execution status."""

    priority: TodoPriority = "medium"
    """Relative importance of this task."""

    created_at: float = field(default_factory=lambda: __import__("time").time())
    """Unix timestamp when the entry was created."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
        }


# Type for todo change callback (async coroutine)
TodoChangeCallback = Callable[["TodoTracker"], Coroutine[Any, Any, None]]


@dataclass
class TodoTracker:
    """Tracks todo/plan entries at the pool level.

    Provides a central place to manage todos that persists across
    agent runs and is accessible from any toolset or endpoint.

    Example:
        ```python
        tracker = TodoTracker()

        # Add entries
        tracker.add("Implement feature X", priority="high")
        tracker.add("Write tests", priority="medium")

        # Update status
        tracker.update_status("todo_1", "in_progress")

        # Get current entries
        for entry in tracker.entries:
            print(f"{entry.status}: {entry.content}")

        # Subscribe to changes
        tracker.on_change = lambda t: print(f"Todos changed: {len(t.entries)} items")
        ```
    """

    entries: list[TodoEntry] = field(default_factory=list)
    """List of all todo entries."""

    _id_counter: int = field(default=0, repr=False)
    """Counter for generating unique IDs."""

    on_change: TodoChangeCallback | None = field(default=None, repr=False)
    """Optional async callback invoked when todos change."""

    _pending_tasks: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    """Track pending notification tasks to prevent garbage collection."""

    def _notify_change(self) -> None:
        """Notify listener of changes (schedules async callback)."""
        if self.on_change is not None:
            task: asyncio.Task[None] = asyncio.create_task(self.on_change(self))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    def _next_id(self) -> str:
        """Generate next unique ID."""
        self._id_counter += 1
        return f"todo_{self._id_counter}"

    def add(
        self,
        content: str,
        *,
        priority: TodoPriority = "medium",
        status: TodoStatus = "pending",
        index: int | None = None,
    ) -> TodoEntry:
        """Add a new todo entry.

        Args:
            content: Description of the task
            priority: Relative importance (high/medium/low)
            status: Initial status (default: pending)
            index: Optional position to insert at (default: append)

        Returns:
            The created TodoEntry
        """
        entry = TodoEntry(
            id=self._next_id(),
            content=content,
            priority=priority,
            status=status,
        )
        if index is None or index >= len(self.entries):
            self.entries.append(entry)
        else:
            self.entries.insert(max(0, index), entry)
        self._notify_change()
        return entry

    def get(self, entry_id: str) -> TodoEntry | None:
        """Get entry by ID.

        Args:
            entry_id: The entry ID to find

        Returns:
            The entry if found, None otherwise
        """
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_by_index(self, index: int) -> TodoEntry | None:
        """Get entry by index.

        Args:
            index: The 0-based index

        Returns:
            The entry if found, None otherwise
        """
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        status: TodoStatus | None = None,
        priority: TodoPriority | None = None,
    ) -> bool:
        """Update an existing entry.

        Args:
            entry_id: The entry ID to update
            content: New content (if provided)
            status: New status (if provided)
            priority: New priority (if provided)

        Returns:
            True if entry was found and updated, False otherwise
        """
        entry = self.get(entry_id)
        if entry is None:
            return False

        changed = False
        if content is not None and entry.content != content:
            entry.content = content
            changed = True
        if status is not None and entry.status != status:
            entry.status = status
            changed = True
        if priority is not None and entry.priority != priority:
            entry.priority = priority
            changed = True
        if changed:
            self._notify_change()
        return True

    def update_by_index(
        self,
        index: int,
        *,
        content: str | None = None,
        status: TodoStatus | None = None,
        priority: TodoPriority | None = None,
    ) -> bool:
        """Update an entry by index.

        Args:
            index: The 0-based index
            content: New content (if provided)
            status: New status (if provided)
            priority: New priority (if provided)

        Returns:
            True if entry was found and updated, False otherwise
        """
        entry = self.get_by_index(index)
        if entry is None:
            return False

        changed = False
        if content is not None and entry.content != content:
            entry.content = content
            changed = True
        if status is not None and entry.status != status:
            entry.status = status
            changed = True
        if priority is not None and entry.priority != priority:
            entry.priority = priority
            changed = True
        if changed:
            self._notify_change()
        return True

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID.

        Args:
            entry_id: The entry ID to remove

        Returns:
            True if entry was found and removed, False otherwise
        """
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                self.entries.pop(i)
                self._notify_change()
                return True
        return False

    def remove_by_index(self, index: int) -> TodoEntry | None:
        """Remove an entry by index.

        Args:
            index: The 0-based index

        Returns:
            The removed entry if found, None otherwise
        """
        if 0 <= index < len(self.entries):
            entry = self.entries.pop(index)
            self._notify_change()
            return entry
        return None

    def clear(self) -> None:
        """Clear all entries."""
        if self.entries:
            self.entries.clear()
            self._notify_change()

    def replace_all(
        self,
        entries: list[tuple[str, TodoPriority, TodoStatus]],
    ) -> None:
        """Replace all entries with new ones (single notification).

        More efficient than clear() + multiple add() calls since it only
        triggers one change notification.

        Args:
            entries: List of (content, priority, status) tuples
        """
        self.entries.clear()
        for content, priority, status in entries:
            entry = TodoEntry(
                id=self._next_id(),
                content=content,
                priority=priority,
                status=status,
            )
            self.entries.append(entry)
        self._notify_change()

    def get_by_status(self, status: TodoStatus) -> list[TodoEntry]:
        """Get all entries with a specific status.

        Args:
            status: The status to filter by

        Returns:
            List of matching entries
        """
        return [e for e in self.entries if e.status == status]

    def to_list(self) -> list[dict[str, Any]]:
        """Convert to list of dicts for JSON serialization."""
        return [e.to_dict() for e in self.entries]
