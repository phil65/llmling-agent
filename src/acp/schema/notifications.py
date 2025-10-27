"""Notification schema definitions."""

from __future__ import annotations

from acp.schema.base import AnnotatedObject
from acp.schema.session_updates import SessionUpdate


class SessionNotification[TSessionUpdate: SessionUpdate = SessionUpdate](AnnotatedObject):
    """Notification containing a session update from the agent.

    Used to stream real-time progress and results during prompt processing.

    See protocol docs: [Agent Reports Output](https://agentclientprotocol.com/protocol/prompt-turn#3-agent-reports-output)
    """

    session_id: str
    """The ID of the session this update pertains to."""

    update: TSessionUpdate


class CancelNotification(AnnotatedObject):
    """Notification to cancel ongoing operations for a session.

    See protocol docs: [Cancellation](https://agentclientprotocol.com/protocol/prompt-turn#cancellation)
    """

    session_id: str
    """The ID of the session to cancel operations for."""
