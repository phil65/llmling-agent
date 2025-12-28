"""Global routes (health, events)."""

from __future__ import annotations

import asyncio
import logging
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


logger = logging.getLogger(__name__)
router = APIRouter(tags=["global"])

VERSION = "0.1.0"


@router.get("/global/health")
async def get_health() -> HealthResponse:
    """Get server health status."""
    return HealthResponse(healthy=True, version=VERSION)


def _serialize_event(event: Event, wrap_payload: bool = False) -> str:
    """Serialize event, optionally wrapping in payload structure."""
    import json

    event_data = event.model_dump(by_alias=True, exclude_none=True)
    if wrap_payload:
        return json.dumps({"payload": event_data})
    return json.dumps(event_data)


async def _event_generator(
    state: ServerState, *, wrap_payload: bool = False
) -> AsyncGenerator[dict[str, Any]]:
    """Generate SSE events."""
    queue: asyncio.Queue[Event] = asyncio.Queue()
    state.event_subscribers.append(queue)
    subscriber_count = len(state.event_subscribers)
    logger.info("SSE: New client connected (total subscribers: %s)", subscriber_count)
    try:
        # Send initial connected event
        connected = ServerConnectedEvent()
        data = _serialize_event(connected, wrap_payload=wrap_payload)
        logger.info("SSE: Sending connected event: %s", data)
        yield {"data": data}
        # Stream events
        while True:
            event = await queue.get()
            data = _serialize_event(event, wrap_payload=wrap_payload)
            logger.info("SSE: Sending event: %s", event.type)
            yield {"data": data}
    finally:
        state.event_subscribers.remove(queue)
        logger.info(
            "SSE: Client disconnected (remaining subscribers: %s)", len(state.event_subscribers)
        )


@router.get("/global/event")
async def get_global_events(state: StateDep) -> EventSourceResponse:
    """Get global events as SSE stream (uses payload wrapper)."""
    return EventSourceResponse(_event_generator(state, wrap_payload=True), sep="\n")


@router.get("/event")
async def get_events(state: StateDep) -> EventSourceResponse:
    """Get events as SSE stream (no payload wrapper)."""
    return EventSourceResponse(_event_generator(state, wrap_payload=False), sep="\n")
