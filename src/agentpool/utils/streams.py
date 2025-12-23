"""Stream utilities for merging async iterators."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def merge_queue_into_iterator[T, V](
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

    # Task to read from primary stream and put into merged queue
    async def primary_task() -> None:
        nonlocal primary_exception
        try:
            async for event in primary_stream:
                await event_queue.put(event)
        except BaseException as e:  # noqa: BLE001
            primary_exception = e
        finally:
            primary_done.set()

    # Task to read from secondary queue and put into merged queue
    async def secondary_task() -> None:
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
            # Now signal end of all events
            await event_queue.put(None)
        except asyncio.CancelledError:
            # Still need to signal completion on cancel
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
        # Clean up tasks
        secondary_task_obj.cancel()
        await asyncio.gather(primary_task_obj, secondary_task_obj, return_exceptions=True)
