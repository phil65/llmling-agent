"""Models for ACP debug server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class DebugSession:
    """Debug session data."""

    session_id: str
    created_at: float
    cwd: str


@dataclass
class NotificationRecord:
    """Record of a sent notification."""

    notification_type: str
    session_id: str
    timestamp: float


@dataclass
class DebugState:
    """Type-safe debug server state."""

    sessions: dict[str, DebugSession] = field(default_factory=dict)
    active_session_id: str | None = None
    notifications_sent: list[NotificationRecord] = field(default_factory=list)
    client_connection: Any = None


# FastAPI models for web interface
class NotificationRequest(BaseModel):
    """Request to send a notification."""

    session_id: str = Field(description="Target session ID")
    notification_type: str = Field(description="Type of notification to send")
    data: dict[str, Any] = Field(default_factory=dict, description="Notification data")


class DebugStatus(BaseModel):
    """Current debug server status."""

    active_sessions: list[str]
    current_session: str | None
    notifications_sent: int
    acp_connected: bool
