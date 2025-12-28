"""Server state management."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import asyncio

    from agentpool.agents.base_agent import BaseAgent
    from agentpool_server.opencode_server.models import (
        Event,
        MessageWithParts,
        Session,
        SessionStatus,
        Todo,
    )


@dataclass
class ServerState:
    """Shared state for the OpenCode server.

    This holds all in-memory state for sessions, messages, etc.
    In the future, this can be backed by proper storage.
    """

    working_dir: str
    start_time: float = field(default_factory=time.time)

    # Session storage
    sessions: dict[str, Session] = field(default_factory=dict)
    session_status: dict[str, SessionStatus] = field(default_factory=dict)

    # Message storage (session_id -> messages)
    messages: dict[str, list[MessageWithParts]] = field(default_factory=dict)

    # Todo storage (session_id -> todos)
    todos: dict[str, list[Todo]] = field(default_factory=dict)

    # SSE event subscribers
    event_subscribers: list[asyncio.Queue[Event]] = field(default_factory=list)

    # AgentPool agent for handling messages
    agent: BaseAgent[Any, Any] | None = None

    async def broadcast_event(self, event: Event) -> None:
        """Broadcast an event to all SSE subscribers."""
        print(f"Broadcasting event: {event.type} to {len(self.event_subscribers)} subscribers")
        for queue in self.event_subscribers:
            await queue.put(event)
