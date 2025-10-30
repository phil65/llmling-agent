"""Task Supervisor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from llmling_agent import log


if TYPE_CHECKING:
    from collections.abc import Coroutine


__all__ = ["TaskSupervisor"]

logger = log.get_logger(__name__)
ErrorHandler = Callable[[asyncio.Task[Any], BaseException], None]


class TaskSupervisor:
    """Track background tasks and provide graceful shutdown semantics.

    Inspired by fasta2a's task manager, this supervisor keeps a registry of
    asyncio tasks created for request handling so they can be cancelled and
    awaited reliably when the connection closes.
    """

    def __init__(self, *, source: str) -> None:
        self._source = source
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._error_handlers: list[ErrorHandler] = []

    def add_error_handler(self, handler: ErrorHandler) -> None:
        self._error_handlers.append(handler)

    def create(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        on_error: ErrorHandler | None = None,
    ) -> asyncio.Task[Any]:
        if self._closed:
            msg = f"TaskSupervisor for {self._source} already closed"
            raise RuntimeError(msg)
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._on_done(t, on_error))
        return task

    def _on_done(self, task: asyncio.Task[Any], on_error: ErrorHandler | None) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            handled = False
            if on_error is not None:
                try:
                    on_error(task, exc)
                    handled = True
                except Exception:
                    logger.exception(
                        "Error in task-specific error handler",
                        source=self._source,
                    )
            if not handled:
                for handler in self._error_handlers:
                    try:
                        handler(task, exc)
                        handled = True
                    except Exception:
                        msg = "Error in supervisor error handler"
                        logger.exception(msg, source=self._source)
            if not handled:
                logger.exception("Unhandled error in task", source=self._source)

    async def shutdown(self) -> None:
        self._closed = True
        if not self._tasks:
            return
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
