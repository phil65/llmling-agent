"""Server state management."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import asyncio

    from agentpool import AgentPool
    from agentpool.agents.base_agent import BaseAgent
    from agentpool_server.opencode_server.input_provider import OpenCodeInputProvider
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

    Uses AgentPool for session persistence and storage.
    In-memory state tracks active sessions and runtime data.
    """

    working_dir: str
    pool: AgentPool[Any]
    agent: BaseAgent[Any, Any]
    start_time: float = field(default_factory=time.time)

    # Active sessions cache (session_id -> OpenCode Session model)
    # This is a cache of sessions loaded from pool.sessions
    sessions: dict[str, Session] = field(default_factory=dict)
    session_status: dict[str, SessionStatus] = field(default_factory=dict)

    # Message storage (session_id -> messages)
    # Runtime cache - messages are also persisted via pool.storage
    messages: dict[str, list[MessageWithParts]] = field(default_factory=dict)

    # Todo storage (session_id -> todos)
    # Uses pool.todos for persistence
    todos: dict[str, list[Todo]] = field(default_factory=dict)

    # Input providers for permission handling (session_id -> provider)
    input_providers: dict[str, OpenCodeInputProvider] = field(default_factory=dict)

    # SSE event subscribers
    event_subscribers: list[asyncio.Queue[Event]] = field(default_factory=list)

    async def broadcast_event(self, event: Event) -> None:
        """Broadcast an event to all SSE subscribers."""
        print(f"Broadcasting event: {event.type} to {len(self.event_subscribers)} subscribers")
        for queue in self.event_subscribers:
            await queue.put(event)
