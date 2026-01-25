"""Thread helpers with free-threading support detection.

This module provides utilities that automatically use true parallelism
on free-threaded Python builds (3.13+ with GIL disabled) and fall back
to sequential or I/O-only concurrent execution on standard builds.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import sys
from typing import TYPE_CHECKING, ParamSpec, TypeVar


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

# Detect free-threading support once at import time
# sys._is_gil_enabled() returns False when GIL is disabled (free-threaded mode)
FREE_THREADED = getattr(sys, "_is_gil_enabled", lambda: True)() is False


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
) -> list[R]:
    """Map function over items, parallelizing on free-threaded builds.

    On free-threaded Python (3.13+ with GIL disabled), uses ThreadPoolExecutor
    for true parallel execution. On standard builds, executes sequentially
    to avoid threading overhead.

    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        max_workers: Maximum worker threads (only used on free-threaded builds)

    Returns:
        List of results in same order as input items
    """
    if FREE_THREADED:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, items))
    return [func(item) for item in items]


def parallel_starmap[R](
    func: Callable[..., R],
    items: Iterable[tuple[object, ...]],
    *,
    max_workers: int | None = None,
) -> list[R]:
    """Like parallel_map but unpacks arguments from tuples.

    Args:
        func: Function to apply to each unpacked tuple
        items: Iterable of argument tuples
        max_workers: Maximum worker threads (only used on free-threaded builds)

    Returns:
        List of results in same order as input items
    """
    if FREE_THREADED:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda args: func(*args), items))
    return [func(*args) for args in items]


async def async_parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
) -> list[R]:
    """Async version of parallel_map using run_in_executor.

    On free-threaded builds, uses a ThreadPoolExecutor for true parallelism.
    On standard builds, still uses executor but parallelism is limited by GIL.

    For I/O-bound work on standard builds, consider using asyncio.gather()
    with async functions instead.

    Args:
        func: Sync function to apply to each item
        items: Iterable of items to process
        max_workers: Maximum worker threads

    Returns:
        List of results in same order as input items
    """
    loop = asyncio.get_running_loop()
    items_list = list(items)

    if FREE_THREADED:
        # True parallelism available
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [loop.run_in_executor(executor, func, item) for item in items_list]
            return list(await asyncio.gather(*futures))
    # GIL limits parallelism, but still useful for releasing control
    # Use default executor to avoid overhead of creating new one
    futures = [loop.run_in_executor(None, func, item) for item in items_list]
    return list(await asyncio.gather(*futures))


async def gather_with_concurrency[T](
    coros: Iterable[Awaitable[T]],
    *,
    limit: int | None = None,
) -> list[T]:
    """Run awaitables with optional concurrency limit.

    Unlike parallel_map, this is for async/await code (I/O-bound work)
    and works the same regardless of free-threading support.

    Args:
        coros: Iterable of awaitables to execute
        limit: Maximum concurrent tasks (None = unlimited)

    Returns:
        List of results in same order as input
    """
    coros_list = list(coros)

    if limit is None or limit <= 0:
        return list(await asyncio.gather(*coros_list))

    semaphore = asyncio.Semaphore(limit)

    async def limited(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*[limited(c) for c in coros_list]))


def parallel_if_free_threaded[**P, R](
    func: Callable[P, R],
) -> Callable[P, R]:
    """Decorator that marks a function for potential parallelization.

    This is a no-op decorator that serves as documentation and could be
    extended in the future to automatically parallelize marked functions.

    Currently just returns the function unchanged but indicates the function
    is safe for parallel execution on free-threaded builds.
    """
    # For now, just mark it - could be extended later
    func._parallel_safe = True  # type: ignore[attr-defined]
    return func


def run_in_thread[**P, R](
    func: Callable[P, R],
) -> Callable[P, Awaitable[R]]:
    """Decorator to run a sync function in a thread pool.

    Useful for wrapping blocking I/O operations for use in async code.

    Args:
        func: Synchronous function to wrap

    Returns:
        Async function that runs the original in a thread
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return wrapper


__all__ = [
    "FREE_THREADED",
    "async_parallel_map",
    "gather_with_concurrency",
    "parallel_if_free_threaded",
    "parallel_map",
    "parallel_starmap",
    "run_in_thread",
]
