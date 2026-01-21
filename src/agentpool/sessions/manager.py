"""Session manager for pool-level session lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from agentpool.log import get_logger
from agentpool.sessions.models import SessionData
from agentpool.sessions.store import MemorySessionStore


if TYPE_CHECKING:
    from types import TracebackType

    from agentpool.delegation.pool import AgentPool
    from agentpool.sessions.store import SessionStore
    from agentpool.storage.manager import StorageManager

logger = get_logger(__name__)


class SessionManager:
    """Manages session data persistence at the pool level.

    Handles:
    - Session data creation and persistence
    - Session listing and retrieval from storage
    - Cleanup of expired sessions

    Note: Runtime session state is managed by agents directly.
    This manager handles only the persistence layer for SessionData.
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        store: SessionStore | None = None,
        pool_id: str | None = None,
        storage: StorageManager | None = None,
    ) -> None:
        """Initialize session manager.

        Args:
            pool: Agent pool for agent access
            store: Session persistence backend (defaults to MemorySessionStore)
            pool_id: Optional identifier for this pool (for multi-pool setups)
            storage: Optional storage manager for project tracking
        """
        self._pool = pool
        self._store = store or MemorySessionStore()
        self._pool_id = pool_id
        self._storage = storage
        logger.debug("Initialized session manager", pool_id=pool_id)

    @property
    def pool(self) -> AgentPool[Any]:
        """Get the agent pool."""
        return self._pool

    @property
    def store(self) -> SessionStore:
        """Get the session store."""
        return self._store

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
        """Exit context and clean up."""
        await self._store.__aexit__(exc_type, exc_val, exc_tb)
        logger.debug("Session manager closed")

    def generate_session_id(self) -> str:
        """Generate a unique, chronologically sortable session ID.

        Uses OpenCode-compatible format: ses_{hex_timestamp}{random_base62}
        IDs are lexicographically sortable by creation time.
        """
        from agentpool.utils.identifiers import generate_session_id

        return generate_session_id()

    async def create(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionData:
        """Create and persist a new session.

        Args:
            agent_name: Name of the initial agent
            session_id: Optional specific session ID (generated if None)
            cwd: Working directory for the session
            metadata: Optional session metadata

        Returns:
            Created session data

        Raises:
            ValueError: If session_id already exists
            KeyError: If agent_name not found in pool
        """
        # Validate agent exists
        if agent_name not in self._pool.all_agents:
            raise KeyError(f"Agent '{agent_name}' not found in pool")

        # Generate session ID if not provided
        if session_id is None:
            session_id = self.generate_session_id()
        else:
            # Check if session already exists
            existing = await self._store.load(session_id)
            if existing is not None:
                raise ValueError(f"Session '{session_id}' already exists")

        # Get or create project if cwd provided and storage available
        project_id: str | None = None
        if cwd and self._storage:
            try:
                from agentpool_storage.project_store import ProjectStore

                project_store = ProjectStore(self._storage)
                project = await project_store.get_or_create(cwd)
                project_id = project.project_id
                logger.debug(
                    "Associated session with project",
                    session_id=session_id,
                    project_id=project_id,
                    worktree=project.worktree,
                )
            except Exception:
                logger.exception("Failed to create/get project for session")

        # Create session data
        data = SessionData(
            session_id=session_id,
            agent_name=agent_name,
            pool_id=self._pool_id,
            project_id=project_id,
            cwd=cwd,
            metadata=metadata or {},
        )

        # Persist to store
        await self._store.save(data)

        logger.info(
            "Created session",
            session_id=session_id,
            agent=agent_name,
        )
        return data

    async def get(self, session_id: str) -> SessionData | None:
        """Get session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data if found, None otherwise
        """
        return await self._store.load(session_id)

    async def save(self, data: SessionData) -> None:
        """Save session data to store.

        Args:
            data: Session data to persist
        """
        await self._store.save(data)

    async def delete(self, session_id: str) -> bool:
        """Delete a session from store.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        return await self._store.delete(session_id)

    async def list_sessions(
        self,
        *,
        agent_name: str | None = None,
    ) -> list[str]:
        """List session IDs.

        Args:
            agent_name: Filter by agent name

        Returns:
            List of session IDs
        """
        return await self._store.list_sessions(
            pool_id=self._pool_id,
            agent_name=agent_name,
        )

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove expired sessions from store.

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions removed
        """
        return await self._store.cleanup_expired(max_age_hours)
