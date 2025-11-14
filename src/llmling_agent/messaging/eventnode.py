"""Event source implementation."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import TYPE_CHECKING, Any, Self


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from types import TracebackType


class Event[TEventData]:
    """Base class for event implementations.

    Handles monitoring for and converting specific types of events.
    Generically typed with the type of event data produced.
    """

    def __init__(self):
        self._stop_event: asyncio.Event = asyncio.Event()

    async def __aenter__(self) -> Self:
        """Set up event resources."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Clean up event resources."""
        self._stop_event.set()

    @abstractmethod
    def create_monitor(self) -> AsyncGenerator[Any]:
        """Create async generator that yields raw event data.

        Yields:
            Raw event data that will be passed to convert_data
        """
        raise NotImplementedError

    @abstractmethod
    async def convert_data(self, raw_data: Any) -> TEventData:
        """Convert raw event data to typed event data.

        Args:
            raw_data: Data from create_monitor

        Returns:
            Typed event data
        """
        raise NotImplementedError
