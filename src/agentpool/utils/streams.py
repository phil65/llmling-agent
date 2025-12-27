"""Stream utilities for merging async iterators."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from agentpool.common_types import SimpleJsonType


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
                    secondary_event = await asyncio.wait_for(secondary_queue.get(), timeout=0.1)
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


def extract_file_path_from_tool_call(tool_name: str, raw_input: dict[str, Any]) -> str | None:
    """Extract file path from a tool call if it's a file-writing tool.

    Uses simple heuristics:
    - Tool name contains 'write' or 'edit' (case-insensitive)
    - Input contains 'path' or 'file_path' key

    Args:
        tool_name: Name of the tool being called
        raw_input: Tool call arguments

    Returns:
        File path if this is a file-writing tool, None otherwise
    """
    name_lower = tool_name.lower()
    if "write" not in name_lower and "edit" not in name_lower:
        return None

    # Try common path argument names
    for key in ("file_path", "path", "filepath", "filename", "file"):
        if key in raw_input and isinstance(val := raw_input[key], str):
            return val

    return None


@dataclass
class FileTracker:
    """Tracks files modified during a stream of events.

    Example:
        ```python
        file_tracker = FileTracker()
        async for event in file_tracker.track(events):
            yield event

        print(f"Modified files: {file_tracker.touched_files}")
        ```
    """

    touched_files: set[str] = field(default_factory=set)
    """Set of file paths that were modified by tool calls."""

    extractor: Callable[[str, dict[str, Any]], str | None] = extract_file_path_from_tool_call
    """Function to extract file path from tool call. Can be customized."""

    def process_event(self, event: Any) -> None:
        """Process an event and track any file modifications.

        Args:
            event: The event to process (checks for ToolCallStartEvent)
        """
        # Import here to avoid circular imports
        from agentpool.agents.events import ToolCallStartEvent

        if isinstance(event, ToolCallStartEvent) and (
            file_path := self.extractor(event.tool_name or "", event.raw_input or {})
        ):
            self.touched_files.add(file_path)

    def track[T](self, stream: AsyncIterator[T]) -> AsyncIterator[T]:
        """Wrap an async iterator to automatically track file modifications.

        Args:
            stream: The event stream to wrap

        Returns:
            Wrapped async iterator that tracks file modifications

        Example:
            ```python
            async for event in file_tracker.track(events):
                yield event
            ```
        """

        async def wrapped() -> AsyncIterator[T]:
            async for event in stream:
                self.process_event(event)
                yield event

        return wrapped()

    def get_metadata(self) -> SimpleJsonType:
        """Get metadata dict with touched files (if any).

        Returns:
            Dict with 'touched_files' key if files were modified, else empty dict
        """
        if self.touched_files:
            return {"touched_files": sorted(self.touched_files)}
        return {}
