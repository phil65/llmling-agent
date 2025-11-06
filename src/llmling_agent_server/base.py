"""Base server class for LLMLing Agent servers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self


if TYPE_CHECKING:
    from types import TracebackType

    from llmling_agent import AgentPool


class BaseServer:
    """Base class for all LLMLing Agent servers.

    Provides standardized interface for server lifecycle management:
    - async def start() - blocking server execution (implemented by subclasses)
    - def start_background() - non-blocking server start via background task
    - def stop() - stop background server task
    - async with run_context() - automatic server start/stop management
    """

    def __init__(self, pool: AgentPool[Any], *args: Any, **kwargs: Any) -> None:
        """Initialize base server with agent pool."""
        self.pool = pool
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    async def __aenter__(self) -> Self:
        """Enter async context and initialize server resources (pool, etc.)."""
        await self.pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context and cleanup server resources."""
        await self.pool.__aexit__(exc_type, exc_val, exc_tb)

    async def start(self) -> None:
        """Start the server (blocking async - runs until stopped).

        This method must be implemented by subclasses and should run
        the server until it's stopped or encounters an error.
        """
        msg = "Subclasses must implement start()"
        raise NotImplementedError(msg)

    def start_background(self) -> None:
        """Start server in background task (non-blocking).

        Creates a background task that runs start() method.
        Server will run in the background until stop() is called.
        """
        if self._task is not None and not self._task.done():
            msg = "Server is already running in background"
            raise RuntimeError(msg)

        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._run_with_shutdown())

    async def _run_with_shutdown(self) -> None:
        """Internal wrapper that handles shutdown signaling."""
        try:
            await self.start()
        finally:
            self._shutdown_event.set()

    def stop(self) -> None:
        """Stop the background server task (non-blocking)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def wait_until_stopped(self) -> None:
        """Wait until the server stops (either by stop() or natural completion)."""
        if self._task is None:
            return

        # Wait for either task completion or shutdown event
        await self._shutdown_event.wait()

        # Ensure task is cleaned up
        if not self._task.done():
            self._task.cancel()

        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    @property
    def is_running(self) -> bool:
        """Check if server is currently running in background."""
        return self._task is not None and not self._task.done()

    @asynccontextmanager
    async def run_context(self):
        """Async context manager for automatic server start/stop.

        Starts the server in background when entering context,
        automatically stops it when exiting context.

        Example:
            async with server:  # Initialize resources
                async with server.run_context():  # Start server
                    # Server is running in background
                    await do_other_work()
                # Server automatically stopped
        """
        self.start_background()
        try:
            yield
        finally:
            self.stop()
            await self.wait_until_stopped()
