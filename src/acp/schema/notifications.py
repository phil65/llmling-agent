"""Notification schema definitions."""

from __future__ import annotations

from typing import Any

from acp.base import Schema
from acp.schema.session_updates import SessionUpdate


class SessionNotification(Schema):
    """Notification containing a session update from the agent.

    Used to stream real-time progress and results during prompt processing.

    See protocol docs: [Agent Reports Output](https://agentclientprotocol.com/protocol/prompt-turn#3-agent-reports-output)
    """

    field_meta: Any | None = None
    """Extension point for implementations."""

    session_id: str
    """The ID of the session this update pertains to."""

    update: SessionUpdate


class CancelNotification(Schema):
    """Notification to cancel ongoing operations for a session.

    See protocol docs: [Cancellation](https://agentclientprotocol.com/protocol/prompt-turn#cancellation)
    """

    field_meta: Any | None = None
    """Extension point for implementations."""

    session_id: str
    """The ID of the session to cancel operations for."""
