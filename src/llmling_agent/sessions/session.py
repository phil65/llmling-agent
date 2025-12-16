"""Protocol-agnostic client session."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from llmling_agent.log import get_logger
from llmling_agent.messaging.message_history import MessageHistory


if TYPE_CHECKING:
    from types import TracebackType

    from llmling_agent.agents.agent import Agent
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.sessions.manager import SessionManager
    from llmling_agent.sessions.models import SessionData

logger = get_logger(__name__)


class ClientSession:
    """Protocol-agnostic runtime session.

    Base class for protocol-specific sessions (ACP, Web, etc.).
    Manages the runtime state of an active session, including:
    - Reference to session data (persistable state)
    - Active agent instance
    - Conversation history (owned by session, not agent)
    - Pool access for agent switching

    The session owns the conversation history and passes it to the agent
    on each run. This makes the agent stateless from the session's perspective.

    Subclasses add protocol-specific functionality:
    - ACPSession: ACP connection, capabilities, slash commands
    - WebSession: WebSocket handling, HTTP session state
    """

    def __init__(
        self,
        data: SessionData,
        pool: AgentPool[Any],
        manager: SessionManager | None = None,
    ) -> None:
        """Initialize client session.

        Args:
            data: Persistable session state
            pool: Agent pool for agent access and switching
            manager: Optional session manager for lifecycle operations
        """
        self._data = data
        self._pool = pool
        self._manager = manager
        self._agent: Agent[Any, Any] | None = None
        self._closed = False
        self._title_generation_triggered = False
        self._title_task: asyncio.Task[None] | None = None
        # Session owns conversation history - agent is stateless
        self._history = MessageHistory()

        logger.debug(
            "Created client session",
            session_id=data.session_id,
            agent=data.agent_name,
        )

    @property
    def session_id(self) -> str:
        """Get session identifier."""
        return self._data.session_id

    @property
    def data(self) -> SessionData:
        """Get session data (persistable state)."""
        return self._data

    @property
    def pool(self) -> AgentPool[Any]:
        """Get agent pool."""
        return self._pool

    @property
    def agent(self) -> Agent[Any, Any]:
        """Get current active agent, creating if needed."""
        if self._agent is None:
            self._agent = self._pool.get_agent(self._data.agent_name)
        return self._agent

    @property
    def agent_name(self) -> str:
        """Get current agent name."""
        return self._data.agent_name

    @property
    def conversation_id(self) -> str:
        """Get conversation ID for message storage."""
        return self._data.conversation_id

    @property
    def history(self) -> MessageHistory:
        """Get the session's conversation history."""
        return self._history

    @property
    def is_closed(self) -> bool:
        """Check if session is closed."""
        return self._closed

    async def __aenter__(self) -> Self:
        """Enter session context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit session context."""
        await self.close()

    async def run(self, prompt: str, **kwargs: Any) -> ChatMessage[Any]:
        """Run the agent with a prompt, using session's history.

        The session passes its own MessageHistory to the agent, making
        the agent stateless from the session's perspective. Messages
        are automatically added to the session's history.

        Args:
            prompt: User prompt to send to the agent
            **kwargs: Additional arguments passed to agent.run()

        Returns:
            The agent's response message
        """
        result = await self.agent.run(
            prompt,
            message_history=self._history,
            conversation_id=self.conversation_id,
            **kwargs,
        )

        # Trigger title generation after first exchange (fire-and-forget)
        # Note: Message logging is handled by the agent itself via MessageNode.log_message
        if not self._title_generation_triggered and self._pool.storage:
            self._title_generation_triggered = True
            self._title_task = asyncio.create_task(self._generate_title())

        return result

    async def _generate_title(self) -> None:
        """Generate conversation title in the background."""
        if not self._pool.storage:
            return
        try:
            messages = self._history.get_history()
            if messages:
                title = await self._pool.storage.generate_conversation_title(
                    self.conversation_id,
                    messages,
                )
                # Also update SessionData so title is available when listing sessions
                if title and self._manager:
                    self._data = self._data.with_title(title)
                    await self._manager.save(self._data)
        except Exception:
            logger.exception(
                "Failed to generate conversation title",
                conversation_id=self.conversation_id,
            )

    async def switch_agent(self, agent_name: str) -> None:
        """Switch to a different agent.

        The conversation history is preserved across agent switches.

        Args:
            agent_name: Name of agent to switch to

        Raises:
            KeyError: If agent not found in pool
        """
        if agent_name not in self._pool.all_agents:
            msg = f"Agent '{agent_name}' not found in pool"
            raise KeyError(msg)

        self._agent = self._pool.get_agent(agent_name)
        self._data = self._data.with_agent(agent_name)

        # Persist the change
        if self._manager:
            await self._manager.save(self._data)

        logger.info(
            "Switched agent",
            session_id=self.session_id,
            agent=agent_name,
        )

    async def touch(self) -> None:
        """Update last_active timestamp and persist."""
        self._data.touch()
        if self._manager:
            await self._manager.save(self._data)

    async def save(self) -> None:
        """Persist current session state."""
        if self._manager:
            await self._manager.save(self._data)
            logger.debug("Saved session", session_id=self.session_id)

    async def close(self) -> None:
        """Close the session and clean up resources."""
        if self._closed:
            return

        self._closed = True
        self._agent = None
        logger.debug("Closed session", session_id=self.session_id)

    def update_metadata(self, **kwargs: Any) -> None:
        """Update session metadata.

        Args:
            **kwargs: Key-value pairs to add/update in metadata
        """
        self._data = self._data.with_metadata(**kwargs)

    def clear_history(self) -> None:
        """Clear the session's conversation history."""
        self._history.clear()

    def get_history_messages(self) -> list[ChatMessage[Any]]:
        """Get all messages in the session's history."""
        return self._history.get_history()
