"""Event source implementation."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import TYPE_CHECKING, Any, Self

from llmling_agent.messaging import ChatMessage, MessageEmitter
from llmling_agent.talk.stats import MessageStats


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from types import TracebackType

    from llmling_agent.messaging.context import NodeContext
    from llmling_agent_config.mcp_server import MCPServerConfig


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


class EventNode[TEventData](MessageEmitter[None, TEventData]):
    """Base class for event sources.

    An event source monitors for events and emits them as messages.
    Generically typed with the type of event data it produces.
    """

    def __init__(
        self,
        event: Event[TEventData],
        name: str | None = None,
        context: NodeContext | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        description: str | None = None,
    ):
        """Initialize event node.

        Args:
            event: Event implementation
            name: Optional name for this node
            context: Optional node context
            mcp_servers: Optional MCP server configurations
            description: Optional description
        """
        super().__init__(name=name, context=context, description=description)
        self.event = event
        self._running = False

    async def __aenter__(self) -> Self:
        """Initialize event resources and start monitoring."""
        await super().__aenter__()
        await self.event.__aenter__()
        # Start monitoring after everything is initialized
        self.task_manager.create_task(self.start())  # Non-blocking
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Stop monitoring and clean up resources."""
        # First stop monitoring
        await self.stop()
        # Then cleanup in reverse order
        await self.event.__aexit__(exc_type, exc_val, exc_tb)
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def start(self):
        """Start monitoring for events."""
        self._running = True
        try:
            async for data in self.event.create_monitor():
                if not self._running:
                    break
                await self.run(data)
        finally:
            self._running = False

    async def stop(self):
        """Stop monitoring for events."""
        self._running = False

    async def get_stats(self) -> MessageStats:
        """Get message statistics (async version)."""
        messages = await self.get_message_history()
        return MessageStats(messages=messages)

    async def _run(self, *content: Any, **kwargs: Any) -> ChatMessage[TEventData]:
        """Convert event data to message."""
        result = await self.event.convert_data(content[0])
        meta = kwargs.get("metadata", {})
        return ChatMessage(
            content=result,
            role="assistant",
            name=self.name,
            metadata=meta,
        )
