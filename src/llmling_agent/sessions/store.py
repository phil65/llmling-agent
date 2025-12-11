"""Session store protocol and implementations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType

    from llmling_agent.sessions.models import SessionData

logger = get_logger(__name__)


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session persistence backends.

    Implementations handle storing and retrieving SessionData to/from
    various backends (SQL, file, memory, etc.).
    """

    async def __aenter__(self) -> Self:
        """Initialize store resources."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up store resources."""
        ...

    @abstractmethod
    async def save(self, data: SessionData) -> None:
        """Save or update session data.

        Args:
            data: Session data to persist
        """
        ...

    @abstractmethod
    async def load(self, session_id: str) -> SessionData | None:
        """Load session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data if found, None otherwise
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        ...

    @abstractmethod
    async def list_sessions(
        self,
        pool_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[str]:
        """List session IDs, optionally filtered.

        Args:
            pool_id: Filter by pool/manifest ID
            agent_name: Filter by agent name

        Returns:
            List of session IDs
        """
        ...

    @abstractmethod
    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age.

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions removed
        """
        ...


class MemorySessionStore:
    """In-memory session store for testing and development."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    async def save(self, data: SessionData) -> None:
        self._sessions[data.session_id] = data
        logger.debug("Saved session", session_id=data.session_id)

    async def load(self, session_id: str) -> SessionData | None:
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("Deleted session", session_id=session_id)
            return True
        return False

    async def list_sessions(
        self,
        pool_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[str]:
        result = []
        for session_id, data in self._sessions.items():
            if pool_id is not None and data.pool_id != pool_id:
                continue
            if agent_name is not None and data.agent_name != agent_name:
                continue
            result.append(session_id)
        return result

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        from llmling_agent.utils.now import get_now

        now = get_now()
        expired = []
        for session_id, data in self._sessions.items():
            age_hours = (now - data.last_active).total_seconds() / 3600
            if age_hours > max_age_hours:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]

        if expired:
            logger.info("Cleaned up expired sessions", count=len(expired))
        return len(expired)
