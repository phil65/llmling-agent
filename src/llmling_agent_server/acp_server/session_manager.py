"""ACP Session Manager - delegates to pool's SessionManager."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from acp.schema import ClientCapabilities
from llmling_agent.log import get_logger
from llmling_agent.sessions import SessionData
from llmling_agent_server.acp_server.session import ACPSession


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from acp import Client
    from acp.schema import McpServer
    from llmling_agent import AgentPool
    from llmling_agent.sessions import SessionManager
    from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


class ACPSessionManager:
    """Manages ACP sessions, delegating persistence to pool's SessionManager.

    This is a thin coordinator that:
    - Creates and tracks active ACP sessions
    - Delegates session persistence to pool.sessions (SessionManager)
    - Handles ACP-specific session initialization
    """

    def __init__(self, pool: AgentPool[Any]) -> None:
        """Initialize ACP session manager.

        Args:
            pool: Agent pool containing SessionManager for persistence
        """
        self._pool = pool
        self._active: dict[str, ACPSession] = {}
        self._lock = asyncio.Lock()
        self._command_update_task: asyncio.Task[None] | None = None
        logger.info("Initialized ACP session manager")

    @property
    def session_manager(self) -> SessionManager:
        """Get the pool's session manager for persistence."""
        return self._pool.sessions

    async def create_session(
        self,
        default_agent_name: str,
        cwd: str,
        client: Client,
        acp_agent: LLMlingACPAgent,
        mcp_servers: Sequence[McpServer] | None = None,
        session_id: str | None = None,
        client_capabilities: ClientCapabilities | None = None,
    ) -> str:
        """Create a new ACP session.

        Args:
            default_agent_name: Name of the default agent to start with
            cwd: Working directory for the session
            client: ACP client connection
            acp_agent: ACP agent instance
            mcp_servers: Optional MCP server configurations
            session_id: Optional specific session ID (generated if None)
            client_capabilities: Client capabilities for tool registration

        Returns:
            Session ID for the created session
        """
        async with self._lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = self.session_manager.generate_session_id()

            # Check for existing session
            if session_id in self._active:
                logger.warning("Session ID already exists", session_id=session_id)
                msg = f"Session {session_id} already exists"
                raise ValueError(msg)

            # Create and persist session data via pool's SessionManager
            data = SessionData(
                session_id=session_id,
                agent_name=default_agent_name,
                conversation_id=f"conv_{session_id}",
                cwd=cwd,
                metadata={
                    "protocol": "acp",
                    "mcp_server_count": len(mcp_servers) if mcp_servers else 0,
                },
            )
            await self.session_manager.save(data)

            # Create the ACP-specific runtime session
            session = ACPSession(
                session_id=session_id,
                agent_pool=self._pool,
                current_agent_name=default_agent_name,
                cwd=cwd,
                client=client,
                mcp_servers=mcp_servers,
                acp_agent=acp_agent,
                client_capabilities=client_capabilities or ClientCapabilities(),
                manager=self,
            )
            session.register_update_callback(self._on_commands_updated)

            # Initialize MCP servers if any are provided
            await session.initialize_mcp_servers()

            self._active[session_id] = session
            logger.info("Created ACP session", session_id=session_id, agent=default_agent_name)
            return session_id

    def get_session(self, session_id: str) -> ACPSession | None:
        """Get an active session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ACPSession instance or None if not found
        """
        return self._active.get(session_id)

    async def resume_session(
        self,
        session_id: str,
        client: Client,
        acp_agent: LLMlingACPAgent,
        client_capabilities: ClientCapabilities | None = None,
    ) -> ACPSession | None:
        """Resume a session from storage.

        Args:
            session_id: Session identifier
            client: ACP client connection
            acp_agent: ACP agent instance
            client_capabilities: Client capabilities

        Returns:
            Resumed ACPSession if found, None otherwise
        """
        async with self._lock:
            # Check if already active
            if session_id in self._active:
                return self._active[session_id]

            # Try to load from pool's session store
            data = await self.session_manager.store.load(session_id)
            if data is None:
                logger.warning("Session not found in store", session_id=session_id)
                return None

            # Validate agent still exists
            if data.agent_name not in self._pool.all_agents:
                logger.warning(
                    "Session agent no longer exists",
                    session_id=session_id,
                    agent=data.agent_name,
                )
                return None

            # Reconstruct ACP session
            session = ACPSession(
                session_id=session_id,
                agent_pool=self._pool,
                current_agent_name=data.agent_name,
                cwd=data.cwd or "",
                client=client,
                mcp_servers=None,  # MCP servers would need to be re-provided
                acp_agent=acp_agent,
                client_capabilities=client_capabilities or ClientCapabilities(),
                manager=self,
            )
            session.register_update_callback(self._on_commands_updated)

            self._active[session_id] = session
            logger.info("Resumed ACP session", session_id=session_id)
            return session

    async def close_session(self, session_id: str, *, delete: bool = False) -> None:
        """Close and optionally delete a session.

        Args:
            session_id: Session identifier to close
            delete: Whether to also delete from persistent storage
        """
        async with self._lock:
            session = self._active.pop(session_id, None)

        if session:
            await session.close()
            logger.info("Closed ACP session", session_id=session_id)

        if delete:
            await self.session_manager.store.delete(session_id)
            logger.info("Deleted session from store", session_id=session_id)

    async def update_session_agent(self, session_id: str, agent_name: str) -> None:
        """Update the agent for a session and persist.

        Args:
            session_id: Session identifier
            agent_name: New agent name
        """
        session = self._active.get(session_id)
        if not session:
            return

        # Load, update, and save session data
        data = await self.session_manager.store.load(session_id)
        if data:
            updated = data.with_agent(agent_name)
            await self.session_manager.save(updated)

    async def list_sessions(self, *, active_only: bool = False) -> list[str]:
        """List session IDs.

        Args:
            active_only: Only return currently active sessions

        Returns:
            List of session IDs
        """
        if active_only:
            return list(self._active.keys())

        return await self.session_manager.store.list_sessions()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - close all active sessions."""
        async with self._lock:
            sessions = list(self._active.values())
            self._active.clear()

        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.exception("Error closing session", session=session.session_id)

        logger.info("Closed all %d ACP sessions", len(sessions))

    def _on_commands_updated(self) -> None:
        """Handle command updates by notifying all active sessions."""
        task = asyncio.create_task(self._update_all_sessions_commands())
        self._command_update_task = task

    async def _update_all_sessions_commands(self) -> None:
        """Update available commands for all active sessions."""
        async with self._lock:
            for session in self._active.values():
                try:
                    await session.send_available_commands_update()
                except Exception:
                    msg = "Failed to update commands"
                    logger.exception(msg, session_id=session.session_id)
