"""Server state management."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import asyncio

    from agentpool import AgentPool
    from agentpool.agents.base_agent import BaseAgent
    from agentpool.diagnostics.lsp_manager import LSPManager
    from agentpool_server.opencode_server.input_provider import OpenCodeInputProvider
    from agentpool_server.opencode_server.models import (
        Config,
        Event,
        MessageWithParts,
        QuestionInfo,
        Session,
        SessionStatus,
        Todo,
    )

# Type alias for async callback
OnFirstSubscriberCallback = Callable[[], Coroutine[Any, Any, None]]


@dataclass
class PendingQuestion:
    """Pending question awaiting user response."""

    session_id: str
    """Session that owns this question."""

    questions: list[QuestionInfo]
    """Questions to ask."""

    future: asyncio.Future[list[list[str]]]
    """Future that resolves when user answers."""

    tool: dict[str, str] | None = None
    """Optional tool context: {message_id, call_id}."""


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

    # Configuration (mutable runtime config)
    # Initialized after state creation
    config: Config | None = None

    # Active sessions cache (session_id -> OpenCode Session model)
    # This is a cache of sessions loaded from pool.sessions
    sessions: dict[str, Session] = field(default_factory=dict)
    session_status: dict[str, SessionStatus] = field(default_factory=dict)

    # Message storage (session_id -> messages)
    # Runtime cache - messages are also persisted via pool.storage
    messages: dict[str, list[MessageWithParts]] = field(default_factory=dict)

    # Reverted messages storage (session_id -> removed messages)
    # Stores messages removed during revert for unrevert operation
    reverted_messages: dict[str, list[MessageWithParts]] = field(default_factory=dict)

    # Todo storage (session_id -> todos)
    # Uses pool.todos for persistence
    todos: dict[str, list[Todo]] = field(default_factory=dict)

    # Input providers for permission handling (session_id -> provider)
    input_providers: dict[str, OpenCodeInputProvider] = field(default_factory=dict)

    # Question storage (question_id -> pending question info)
    pending_questions: dict[str, PendingQuestion] = field(default_factory=dict)

    # SSE event subscribers
    event_subscribers: list[asyncio.Queue[Event]] = field(default_factory=list)

    # Callback for first subscriber connection (e.g., for update check)
    on_first_subscriber: OnFirstSubscriberCallback | None = None
    _first_subscriber_triggered: bool = field(default=False, repr=False)

    # Background tasks (for cleanup on shutdown)
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    # LSP manager for language server integration (initialized lazily)
    lsp_manager: LSPManager | None = None

    def create_background_task(self, coro: Any, *, name: str | None = None) -> asyncio.Task[Any]:
        """Create and track a background task."""
        import asyncio

        task = asyncio.create_task(coro, name=name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    async def cleanup_tasks(self) -> None:
        """Cancel and wait for all background tasks."""
        for task in self.background_tasks:
            task.cancel()
        if self.background_tasks:
            import asyncio

            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

    async def broadcast_event(self, event: Event) -> None:
        """Broadcast an event to all SSE subscribers."""
        print(f"Broadcasting event: {event.type} to {len(self.event_subscribers)} subscribers")
        for queue in self.event_subscribers:
            await queue.put(event)

    def get_or_create_lsp_manager(self) -> LSPManager:
        """Get or create the LSP manager.

        Creates the LSP manager lazily using the agent's execution environment.

        Returns:
            The LSP manager instance.

        Raises:
            RuntimeError: If the agent doesn't have an execution environment.
        """
        if self.lsp_manager is not None:
            return self.lsp_manager

        from agentpool.diagnostics.lsp_manager import LSPManager

        # Get the execution environment from the agent
        env = getattr(self.agent, "env", None)
        if env is None:
            msg = "Agent does not have an execution environment for LSP"
            raise RuntimeError(msg)

        self.lsp_manager = LSPManager(env=env)
        self.lsp_manager.register_defaults()
        return self.lsp_manager
