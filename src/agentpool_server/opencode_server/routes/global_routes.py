"""Global routes (health, events)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (  # noqa: TC001
    Event,
    HealthResponse,
    ServerConnectedEvent,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from agentpool_server.opencode_server.state import ServerState


router = APIRouter(tags=["global"])

VERSION = "0.1.0"


@router.get("/global/health")
async def get_health() -> HealthResponse:
    """Get server health status."""
    return HealthResponse(healthy=True, version=VERSION)


async def _event_generator(state: ServerState) -> AsyncGenerator[dict[str, Any]]:
    """Generate SSE events."""
    queue: asyncio.Queue[Event] = asyncio.Queue()
    state.event_subscribers.append(queue)
    try:
        # Send initial connected event
        yield {
            "event": "message",
            "data": ServerConnectedEvent().model_dump_json(by_alias=True),
        }
        # Stream events
        while True:
            event = await queue.get()
            yield {
                "event": "message",
                "data": event.model_dump_json(by_alias=True),
            }
    finally:
        state.event_subscribers.remove(queue)


@router.get("/global/event")
async def get_global_events(state: StateDep) -> EventSourceResponse:
    """Get global events as SSE stream."""
    return EventSourceResponse(_event_generator(state))


@router.get("/event")
async def get_events(state: StateDep) -> EventSourceResponse:
    """Get events as SSE stream."""
    return EventSourceResponse(_event_generator(state))
