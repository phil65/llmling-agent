"""Session data models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field
from schemez import Schema

from llmling_agent.utils.now import get_now


class SessionData(Schema):
    """Persistable session state.

    Contains all information needed to persist and restore a session.
    Protocol-specific data (ACP capabilities, web cookies, etc.) goes in metadata.
    """

    session_id: str
    """Unique session identifier."""

    agent_name: str
    """Name of the currently active agent."""

    conversation_id: str
    """Links to conversation in StorageManager."""

    title: str | None = None
    """AI-generated or user-provided title for the conversation."""

    pool_id: str | None = None
    """Optional pool/manifest identifier for multi-pool setups."""

    cwd: str | None = None
    """Working directory for the session."""

    created_at: datetime = Field(default_factory=get_now)
    """When the session was created."""

    last_active: datetime = Field(default_factory=get_now)
    """Last activity timestamp."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Protocol-specific or custom metadata.

    Examples:
        - ACP: client_capabilities, mcp_servers
        - Web: user_id, auth_token
        - CLI: terminal_size, color_support
    """

    def touch(self) -> None:
        """Update last_active timestamp."""
        # Note: Schema is frozen by default, so we need to work around that
        # by using object.__setattr__ or making this field mutable
        object.__setattr__(self, "last_active", get_now())

    def with_agent(self, agent_name: str) -> SessionData:
        """Return copy with different agent."""
        return self.model_copy(update={"agent_name": agent_name, "last_active": get_now()})

    def with_metadata(self, **kwargs: Any) -> SessionData:
        """Return copy with updated metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return self.model_copy(update={"metadata": new_metadata, "last_active": get_now()})

    def with_title(self, title: str) -> SessionData:
        """Return copy with updated title."""
        return self.model_copy(update={"title": title, "last_active": get_now()})
