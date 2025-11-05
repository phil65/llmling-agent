"""ACP Session Manager."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self
import uuid

from llmling_agent.log import get_logger
from llmling_agent_acp.session import ACPSession


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp import Client
    from acp.schema import ClientCapabilities, McpServer
    from llmling_agent import AgentPool
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


class ACPSessionManager:
    """Manages multiple ACP sessions and their lifecycle.

    Provides centralized management of ACP sessions, including:
    - Session creation and initialization
    - Session lookup and retrieval
    - Session cleanup and resource management
    - Agent instance management
    """

    def __init__(self) -> None:
        """Initialize session manager.

        Args:
            command_bridge: Optional command bridge for slash commands
        """
        self._sessions: dict[str, ACPSession] = {}
        self._lock = asyncio.Lock()
        self._command_update_task: asyncio.Task[None] | None = None
        logger.info("Initialized ACP session manager")

    async def create_session(
        self,
        agent_pool: AgentPool[Any],
        default_agent_name: str,
        cwd: str,
        client: Client,  # This is the AgentSideConnection
        acp_agent: LLMlingACPAgent,
        mcp_servers: Sequence[McpServer] | None = None,
        session_id: str | None = None,
        client_capabilities: ClientCapabilities | None = None,
    ) -> str:
        """Create a new ACP session.

        Args:
            agent_pool: AgentPool containing available agents
            default_agent_name: Name of the default agent to start with
            cwd: Working directory for the session
            client: External library Client interface
            mcp_servers: Optional MCP server configurations
            session_id: Optional specific session ID (generated if None)
            acp_agent: ACP agent instance for capability tools
            client_capabilities: Client capabilities for tool registration

        Returns:
            Session ID for the created session
        """
        async with self._lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = f"sess_{uuid.uuid4().hex[:12]}"

            # Check for existing session
            if session_id in self._sessions:
                logger.warning("Session ID already exists", session_id=session_id)
                msg = f"Session {session_id} already exists"
                raise ValueError(msg)

            session = ACPSession(
                session_id=session_id,
                agent_pool=agent_pool,
                current_agent_name=default_agent_name,
                cwd=cwd,
                client=client,
                mcp_servers=mcp_servers,
                acp_agent=acp_agent,
                client_capabilities=client_capabilities,
                manager=self,
            )
            session.register_update_callback(self._on_commands_updated)

            # Initialize MCP servers if any are provided
            #  TODO: perhaps move to ACPAgent.new_session?
            await session.initialize_mcp_servers()

            self._sessions[session_id] = session
            return session_id  # Commands will be sent after session response is returned

    def get_session(self, session_id: str) -> ACPSession | None:
        """Get an existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ACPSession instance or None if not found
        """
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """Close and remove a session.

        Args:
            session_id: Session identifier to close
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                await session.close()
                logger.info("Removed session", session_id=session_id)
            else:
                logger.warning(
                    "Attempted to close non-existent session",
                    session_id=session_id,
                )

    async def cleanup_inactive_sessions(self) -> None:
        """Remove any inactive sessions."""
        async with self._lock:
            inactive_sessions = [
                session_id
                for session_id, session in self._sessions.items()
                if not session.active
            ]

            for session_id in inactive_sessions:
                session = self._sessions.pop(session_id, None)
                if session:
                    try:
                        await session.close()
                    except Exception:
                        logger.exception(
                            "Error closing inactive session",
                            session_id=session_id,
                        )

            if inactive_sessions:
                logger.info("Cleaned up %d inactive sessions", len(inactive_sessions))

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object):
        """Async context manager exit."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        # Close sessions outside of lock to avoid deadlock
        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.exception("Error closing session", session=session.session_id)

        logger.info("Closed all %d sessions", len(sessions))

    def _on_commands_updated(self) -> None:
        """Handle command updates by notifying all active sessions."""
        # Schedule async task to update all sessions
        task = asyncio.create_task(self._update_all_sessions_commands())
        # Store reference to prevent garbage collection
        self._command_update_task = task

    async def _update_all_sessions_commands(self) -> None:
        """Update available commands for all active sessions."""
        async with self._lock:
            for session in self._sessions.values():
                try:
                    await session.send_available_commands_update()
                except Exception:
                    msg = "Failed to update commands"
                    logger.exception(msg, session_id=session.session_id)
