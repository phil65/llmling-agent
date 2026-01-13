"""Session protocol for unified session management across agent types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionInfo(Protocol):
    """Protocol for session information.

    This protocol provides a unified interface for session metadata across
    different agent implementations (ACP, ClaudeCode, native agents).

    Both ACP's SessionInfo and our SessionData can fulfill this protocol.
    """

    session_id: str
    """Unique identifier for the session."""

    cwd: str | None
    """Working directory for the session (absolute path)."""

    title: str | None
    """Human-readable title for the session."""

    updated_at: str | None
    """ISO 8601 timestamp of last activity."""
