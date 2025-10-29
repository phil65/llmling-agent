"""Stream utilities for merging async iterators."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator

T = TypeVar("T")


@asynccontextmanager
async def merge_queue_into_iterator[T](
    primary_stream: AsyncIterator[T],
    secondary_queue: asyncio.Queue[Any],
) -> AsyncIterator[AsyncIterator[T | Any]]:
    """Merge a primary async stream with events from a secondary queue.

    Args:
        primary_stream: The main async iterator (e.g., provider events)
        secondary_queue: Queue containing secondary events (e.g., progress events)

    Yields:
        Async iterator that yields events from both sources in real-time

    Example:
        ```python
        progress_queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        async with merge_queue_into_iterator(provider_stream, progress_queue) as events:
            async for event in events:
                print(f"Got event: {event}")
        ```
    """
    # Create a queue for all merged events
    event_queue: asyncio.Queue[T | Any | None] = asyncio.Queue()

    # Task to read from primary stream and put into merged queue
    async def primary_task():
        try:
            async for event in primary_stream:
                await event_queue.put(event)
        finally:
            # Signal end of primary stream
            await event_queue.put(None)

    # Task to read from secondary queue and put into merged queue
    async def secondary_task():
        try:
            while True:
                secondary_event = await secondary_queue.get()
                await event_queue.put(secondary_event)
        except asyncio.CancelledError:
            pass

    # Start both tasks
    primary_task_obj = asyncio.create_task(primary_task())
    secondary_task_obj = asyncio.create_task(secondary_task())

    try:
        # Create async iterator that drains the merged queue
        async def merged_events():
            while True:
                event = await event_queue.get()
                if event is None:  # End of primary stream
                    break
                yield event

        yield merged_events()

    finally:
        # Clean up tasks
        secondary_task_obj.cancel()
        await asyncio.gather(primary_task_obj, secondary_task_obj, return_exceptions=True)


@asynccontextmanager
async def merge_async_iterables[T](
    *iterables: AsyncIterable[T],
) -> AsyncIterator[AsyncIterator[T]]:
    """Merge multiple async iterables into a single stream.

    Args:
        *iterables: Variable number of async iterables to merge

    Yields:
        Async iterator that yields events from all iterables as they arrive

    Example:
        ```python
        async def gen1():
            yield 1
            yield 2

        async def gen2():
            yield "a"
            yield "b"

        async with merge_async_iterables(gen1(), gen2()) as events:
            async for event in events:
                print(f"Got: {event}")  # Prints: 1, "a", 2, "b" (or similar order)
        ```
    """
    if not iterables:
        # Handle empty case
        async def empty():
            return
            yield  # Make it an async generator

        yield empty()
        return

    # Create a queue for all merged events
    event_queue: asyncio.Queue[T | None] = asyncio.Queue()
    completed_count = 0

    # Task for each iterable
    async def iterable_task(iterable: AsyncIterable[T]):
        nonlocal completed_count
        try:
            async for event in iterable:
                await event_queue.put(event)
        finally:
            completed_count += 1
            # Signal completion when all iterables are done
            if completed_count == len(iterables):
                await event_queue.put(None)

    # Start all tasks
    tasks = [asyncio.create_task(iterable_task(iterable)) for iterable in iterables]

    try:
        # Create async iterator that drains the merged queue
        async def merged_events():
            while True:
                event = await event_queue.get()
                if event is None:  # All iterables completed
                    break
                yield event

        yield merged_events()

    finally:
        # Clean up tasks
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
