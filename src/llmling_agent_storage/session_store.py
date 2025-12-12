"""SQL session store implementation."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Self

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from llmling_agent.log import get_logger
from llmling_agent.sessions.models import SessionData
from llmling_agent.utils.now import get_now
from llmling_agent_storage.sql_provider.models import Session


if TYPE_CHECKING:
    from types import TracebackType

    from sqlalchemy.ext.asyncio import AsyncEngine

    from llmling_agent_config.storage import SQLStorageConfig

logger = get_logger(__name__)


class SQLSessionStore:
    """SQL-based session store using SQLModel.

    Persists session data to a SQL database, supporting:
    - SQLite (default)
    - PostgreSQL
    - MySQL
    """

    def __init__(self, config: SQLStorageConfig) -> None:
        """Initialize SQL session store.

        Args:
            config: SQL storage configuration with database URL
        """
        self._config = config
        self._engine: AsyncEngine | None = None

    async def __aenter__(self) -> Self:
        """Initialize database connection and create tables."""
        self._engine = self._config.get_engine()

        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Auto-migrate missing columns
        await self._migrate_columns()

        logger.debug("SQL session store initialized", url=self._config.url)
        return self

    async def _migrate_columns(self) -> None:
        """Add missing columns to existing tables."""
        from llmling_agent_storage.sql_provider.utils import auto_migrate_columns

        if self._engine is None:
            return

        async with self._engine.begin() as conn:

            def sync_migrate(sync_conn: Any) -> None:
                auto_migrate_columns(sync_conn, self._engine.dialect)  # type: ignore[union-attr]

            await conn.run_sync(sync_migrate)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    def _get_engine(self) -> AsyncEngine:
        """Get engine, raising if not initialized."""
        if self._engine is None:
            msg = "Session store not initialized. Use async context manager."
            raise RuntimeError(msg)
        return self._engine

    def _to_db_model(self, data: SessionData) -> Session:
        """Convert SessionData to database model."""
        return Session(
            session_id=data.session_id,
            agent_name=data.agent_name,
            conversation_id=data.conversation_id,
            pool_id=data.pool_id,
            title=data.title,
            cwd=data.cwd,
            created_at=data.created_at,
            last_active=data.last_active,
            metadata_json=data.metadata,
        )

    def _from_db_model(self, row: Session) -> SessionData:
        """Convert database model to SessionData."""
        return SessionData(
            session_id=row.session_id,
            agent_name=row.agent_name,
            conversation_id=row.conversation_id,
            pool_id=row.pool_id,
            title=row.title,
            cwd=row.cwd,
            created_at=row.created_at,
            last_active=row.last_active,
            metadata=row.metadata_json or {},
        )

    async def save(self, data: SessionData) -> None:
        """Save or update session data.

        Uses delete-then-insert for upsert semantics.

        Args:
            data: Session data to persist
        """
        engine = self._get_engine()

        async with AsyncSession(engine) as session:
            # Delete existing if present (upsert via delete+insert)
            stmt = delete(Session).where(Session.session_id == data.session_id)  # type: ignore[arg-type]
            await session.execute(stmt)

            # Insert new/updated
            db_session = self._to_db_model(data)
            session.add(db_session)
            await session.commit()
            logger.debug("Saved session", session_id=data.session_id)

    async def load(self, session_id: str) -> SessionData | None:
        """Load session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data if found, None otherwise
        """
        engine = self._get_engine()

        async with AsyncSession(engine) as session:
            stmt = select(Session).where(Session.session_id == session_id)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            row = result.scalars().first()

            if row is None:
                return None

            return self._from_db_model(row)

    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        engine = self._get_engine()

        async with AsyncSession(engine) as session:
            stmt = delete(Session).where(Session.session_id == session_id)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            await session.commit()

            deleted: bool = result.rowcount > 0  # type: ignore[attr-defined]
            if deleted:
                logger.debug("Deleted session", session_id=session_id)
            return deleted

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
        engine = self._get_engine()

        async with AsyncSession(engine) as session:
            stmt = select(Session.session_id)  # type: ignore[call-overload]

            if pool_id is not None:
                stmt = stmt.where(Session.pool_id == pool_id)
            if agent_name is not None:
                stmt = stmt.where(Session.agent_name == agent_name)

            stmt = stmt.order_by(Session.last_active.desc())  # type: ignore[attr-defined]
            result = await session.execute(stmt)
            # When selecting a single column, scalars() gives us the values directly
            return list(result.scalars().all())

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age.

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions removed
        """
        engine = self._get_engine()
        cutoff = get_now() - timedelta(hours=max_age_hours)

        async with AsyncSession(engine) as session:
            stmt = delete(Session).where(Session.last_active < cutoff)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            await session.commit()

            count = result.rowcount or 0  # type: ignore[attr-defined]
            if count > 0:
                logger.info("Cleaned up expired sessions", count=count)
            return count

    async def get_all(
        self,
        pool_id: str | None = None,
        limit: int | None = None,
    ) -> list[SessionData]:
        """Get all sessions, optionally filtered.

        Args:
            pool_id: Filter by pool/manifest ID
            limit: Maximum number of sessions to return

        Returns:
            List of session data objects
        """
        engine = self._get_engine()

        async with AsyncSession(engine) as session:
            stmt = select(Session)

            if pool_id is not None:
                stmt = stmt.where(Session.pool_id == pool_id)  # type: ignore[arg-type]

            stmt = stmt.order_by(Session.last_active.desc())  # type: ignore[attr-defined]

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            # scalars() gives us actual Session model instances
            return [self._from_db_model(row) for row in result.scalars().all()]
