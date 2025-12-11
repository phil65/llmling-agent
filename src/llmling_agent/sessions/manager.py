"""Session manager for pool-level session lifecycle."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from llmling_agent.log import get_logger
from llmling_agent.sessions.models import SessionData
from llmling_agent.sessions.session import ClientSession
from llmling_agent.sessions.store import MemorySessionStore


if TYPE_CHECKING:
    from types import TracebackType

    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.sessions.store import SessionStore

logger = get_logger(__name__)


class SessionManager:
    """Manages session lifecycle at the pool level.

    Handles:
    - Session creation and initialization
    - Active session tracking (in-memory)
    - Session persistence via SessionStore
    - Session resumption from storage
    - Cleanup of expired sessions
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        store: SessionStore | None = None,
        pool_id: str | None = None,
    ) -> None:
        """Initialize session manager.

        Args:
            pool: Agent pool for agent access
            store: Session persistence backend (defaults to MemorySessionStore)
            pool_id: Optional identifier for this pool (for multi-pool setups)
        """
        self._pool = pool
        self._store = store or MemorySessionStore()
        self._pool_id = pool_id
        self._active: dict[str, ClientSession] = {}
        self._lock = asyncio.Lock()
        logger.debug("Initialized session manager", pool_id=pool_id)

    @property
    def pool(self) -> AgentPool[Any]:
        """Get the agent pool."""
        return self._pool

    @property
    def store(self) -> SessionStore:
        """Get the session store."""
        return self._store

    @property
    def active_sessions(self) -> dict[str, ClientSession]:
        """Get currently active sessions (read-only view)."""
        return dict(self._active)

    async def __aenter__(self) -> Self:
        """Enter context and initialize store."""
        await self._store.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context and clean up all sessions."""
        # Close all active sessions
        async with self._lock:
            sessions = list(self._active.values())
            self._active.clear()

        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.exception("Error closing session", session_id=session.session_id)

        await self._store.__aexit__(exc_type, exc_val, exc_tb)
        logger.debug("Session manager closed", session_count=len(sessions))

    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"sess_{uuid4().hex[:12]}"

    async def create(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_class: type[ClientSession] = ClientSession,
        **session_kwargs: Any,
    ) -> ClientSession:
        """Create a new session.

        Args:
            agent_name: Name of the initial agent
            session_id: Optional specific session ID (generated if None)
            cwd: Working directory for the session
            conversation_id: Optional conversation ID (generated if None)
            metadata: Optional session metadata
            session_class: Session class to instantiate (for protocol-specific sessions)
            **session_kwargs: Additional kwargs passed to session constructor

        Returns:
            Created session instance

        Raises:
            ValueError: If session_id already exists
            KeyError: If agent_name not found in pool
        """
        # Validate agent exists
        if agent_name not in self._pool.all_agents:
            msg = f"Agent '{agent_name}' not found in pool"
            raise KeyError(msg)

        async with self._lock:
            # Generate or validate session ID
            if session_id is None:
                session_id = self.generate_session_id()
            elif session_id in self._active:
                msg = f"Session '{session_id}' already exists"
                raise ValueError(msg)

            # Create session data
            data = SessionData(
                session_id=session_id,
                agent_name=agent_name,
                conversation_id=conversation_id or f"conv_{uuid4().hex[:12]}",
                pool_id=self._pool_id,
                cwd=cwd,
                metadata=metadata or {},
            )

            # Persist to store
            await self._store.save(data)

            # Create runtime session
            session = session_class(
                data=data,
                pool=self._pool,
                manager=self,
                **session_kwargs,
            )

            self._active[session_id] = session
            logger.info(
                "Created session",
                session_id=session_id,
                agent=agent_name,
            )
            return session

    async def get(self, session_id: str) -> ClientSession | None:
        """Get an active session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Active session if found, None otherwise
        """
        return self._active.get(session_id)

    async def resume(
        self,
        session_id: str,
        session_class: type[ClientSession] = ClientSession,
        **session_kwargs: Any,
    ) -> ClientSession | None:
        """Resume a session from storage.

        Loads session data from store and creates a runtime session.

        Args:
            session_id: Session identifier
            session_class: Session class to instantiate
            **session_kwargs: Additional kwargs passed to session constructor

        Returns:
            Resumed session if found in store, None otherwise
        """
        async with self._lock:
            # Check if already active
            if session_id in self._active:
                return self._active[session_id]

            # Try to load from store
            data = await self._store.load(session_id)
            if data is None:
                return None

            # Validate agent still exists
            if data.agent_name not in self._pool.all_agents:
                logger.warning(
                    "Session agent no longer exists",
                    session_id=session_id,
                    agent=data.agent_name,
                )
                return None

            # Create runtime session
            session = session_class(
                data=data,
                pool=self._pool,
                manager=self,
                **session_kwargs,
            )

            self._active[session_id] = session
            logger.info("Resumed session", session_id=session_id)
            return session

    async def close(self, session_id: str, *, delete: bool = False) -> bool:
        """Close and optionally delete a session.

        Args:
            session_id: Session identifier
            delete: Whether to also delete from store

        Returns:
            True if session was closed, False if not found
        """
        async with self._lock:
            session = self._active.pop(session_id, None)

        if session:
            await session.close()
            if delete:
                await self._store.delete(session_id)
            logger.info("Closed session", session_id=session_id, deleted=delete)
            return True

        # Session not active, but maybe in store
        if delete:
            return await self._store.delete(session_id)

        return False

    async def save(self, data: SessionData) -> None:
        """Save session data to store.

        Args:
            data: Session data to persist
        """
        await self._store.save(data)

    async def list_sessions(
        self,
        *,
        active_only: bool = False,
        agent_name: str | None = None,
    ) -> list[str]:
        """List session IDs.

        Args:
            active_only: Only return currently active sessions
            agent_name: Filter by agent name

        Returns:
            List of session IDs
        """
        if active_only:
            sessions = list(self._active.keys())
            if agent_name:
                sessions = [sid for sid, s in self._active.items() if s.agent_name == agent_name]
            return sessions

        return await self._store.list_sessions(
            pool_id=self._pool_id,
            agent_name=agent_name,
        )

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove expired sessions from store.

        Does not affect currently active sessions.

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions removed
        """
        return await self._store.cleanup_expired(max_age_hours)
