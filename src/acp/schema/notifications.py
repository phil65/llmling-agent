"""Notification schema definitions."""

from __future__ import annotations

from typing import Any, TypeVar

from acp.schema.base import AnnotatedObject
from acp.schema.session_updates import SessionUpdate


TSessionUpdate_co = TypeVar(
    "TSessionUpdate_co",
    covariant=True,
    bound=SessionUpdate,
)


class SessionNotification[TSessionUpdate_co: SessionUpdate = SessionUpdate](AnnotatedObject):
    """Notification containing a session update from the agent.

    Used to stream real-time progress and results during prompt processing.

    See protocol docs: [Agent Reports Output](https://agentclientprotocol.com/protocol/prompt-turn#3-agent-reports-output)
    """

    session_id: str
    """The ID of the session this update pertains to."""

    update: TSessionUpdate_co
    """The session update data."""


class CancelNotification(AnnotatedObject):
    """Notification to cancel ongoing operations for a session.

    This is a notification sent by the client to cancel an ongoing prompt turn.

    Upon receiving this notification, the Agent SHOULD:
    - Stop all language model requests as soon as possible
    - Abort all tool call invocations in progress
    - Send any pending `session/update` notifications
    - Respond to the original `session/prompt` request with `StopReason::Cancelled`

    See protocol docs: [Cancellation](https://agentclientprotocol.com/protocol/prompt-turn#cancellation)
    """

    session_id: str
    """The ID of the session to cancel operations for."""


class ExtNotification(AnnotatedObject):
    """Extension notification from client or agent.

    Allows sending arbitrary notifications that are not part of the ACP spec.
    Extension notifications provide a way to send one-way messages for custom
    functionality while maintaining protocol compatibility.

    The method name should be prefixed with an underscore (e.g., `_myExtension/notify`).

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    method: str
    """The extension method name (should be prefixed with underscore)."""

    params: dict[str, Any] | None = None
    """Optional parameters for the notification."""


AgentNotification = SessionNotification | ExtNotification
"""All possible notifications that an agent can send to a client.

This is used internally for routing RPC notifications.
Notifications do not expect a response.
"""

ClientNotification = CancelNotification | ExtNotification
"""All possible notifications that a client can send to an agent.

This is used internally for routing RPC notifications.
Notifications do not expect a response.
"""
