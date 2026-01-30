"""Global routes (health, events)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import anyenv
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from agentpool import log
from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import (  # noqa: TC001
    Event,
    HealthResponse,
    ServerConnectedEvent,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from agentpool_server.opencode_server.state import ServerState


logger = log.get_logger(__name__)
router = APIRouter(tags=["global"])

VERSION = "0.1.0"


@router.get("/global/health")
async def get_health() -> HealthResponse:
    """Get server health status."""
    return HealthResponse(healthy=True, version=VERSION)


def _serialize_event(event: Event, wrap_payload: bool = False) -> str:
    """Serialize event, optionally wrapping in payload structure."""
    event_data = event.model_dump(by_alias=True, exclude_none=True)
    if wrap_payload:
        return anyenv.dump_json({"payload": event_data})
    return anyenv.dump_json(event_data)


async def _event_generator(
    state: ServerState, *, wrap_payload: bool = False
) -> AsyncGenerator[dict[str, Any]]:
    """Generate SSE events."""
    queue: asyncio.Queue[Event] = asyncio.Queue()
    state.event_subscribers.append(queue)
    subscriber_count = len(state.event_subscribers)
    logger.info("SSE: New client connected (total subscribers: %s)", subscriber_count)

    # Trigger first subscriber callback if this is the first connection
    if (
        subscriber_count == 1
        and not state._first_subscriber_triggered
        and state.on_first_subscriber is not None
    ):
        state._first_subscriber_triggered = True
        state.create_background_task(state.on_first_subscriber(), name="on_first_subscriber")

    try:
        # Send initial connected event
        connected = ServerConnectedEvent()
        data = _serialize_event(connected, wrap_payload=wrap_payload)
        logger.info("SSE: Sending connected event", data=data)
        yield {"data": data}
        # Stream events
        while True:
            event = await queue.get()
            data = _serialize_event(event, wrap_payload=wrap_payload)
            logger.info("SSE: Sending event", event_type=event.type)
            yield {"data": data}
    finally:
        state.event_subscribers.remove(queue)
        logger.info("SSE: Client disconnected", remaining_subscribers=len(state.event_subscribers))


@router.get("/global/event")
async def get_global_events(state: StateDep) -> EventSourceResponse:
    """Get global events as SSE stream (uses payload wrapper)."""
    return EventSourceResponse(_event_generator(state, wrap_payload=True), sep="\n")


@router.get("/event")
async def get_events(state: StateDep) -> EventSourceResponse:
    """Get events as SSE stream (no payload wrapper)."""
    return EventSourceResponse(_event_generator(state, wrap_payload=False), sep="\n")
