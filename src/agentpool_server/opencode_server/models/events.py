"""SSE event models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.message import (  # noqa: TC001
    AssistantMessage,
    UserMessage,
)
from agentpool_server.opencode_server.models.parts import Part  # noqa: TC001
from agentpool_server.opencode_server.models.session import (  # noqa: TC001
    Session,
    SessionStatus,
)


class EmptyProperties(OpenCodeBaseModel):
    """Empty properties object."""


class ServerConnectedEvent(OpenCodeBaseModel):
    """Server connected event."""

    type: Literal["server.connected"] = "server.connected"
    properties: EmptyProperties = Field(default_factory=EmptyProperties)


class SessionInfoProperties(OpenCodeBaseModel):
    """Session info wrapper for events."""

    info: Session


class SessionCreatedEvent(OpenCodeBaseModel):
    """Session created event."""

    type: Literal["session.created"] = "session.created"
    properties: SessionInfoProperties


class SessionUpdatedEvent(OpenCodeBaseModel):
    """Session updated event."""

    type: Literal["session.updated"] = "session.updated"
    properties: SessionInfoProperties


class SessionDeletedProperties(OpenCodeBaseModel):
    """Properties for session deleted event."""

    session_id: str


class SessionDeletedEvent(OpenCodeBaseModel):
    """Session deleted event."""

    type: Literal["session.deleted"] = "session.deleted"
    properties: SessionDeletedProperties


class SessionStatusProperties(OpenCodeBaseModel):
    """Properties for session status event."""

    session_id: str
    status: SessionStatus


class SessionStatusEvent(OpenCodeBaseModel):
    """Session status event."""

    type: Literal["session.status"] = "session.status"
    properties: SessionStatusProperties


class SessionIdleProperties(OpenCodeBaseModel):
    """Properties for session idle event (deprecated but still used by TUI)."""

    session_id: str


class SessionIdleEvent(OpenCodeBaseModel):
    """Session idle event (deprecated but still used by TUI run command)."""

    type: Literal["session.idle"] = "session.idle"
    properties: SessionIdleProperties


class MessageUpdatedEventProperties(OpenCodeBaseModel):
    """Properties for message updated event."""

    info: UserMessage | AssistantMessage


class MessageUpdatedEvent(OpenCodeBaseModel):
    """Message updated event."""

    type: Literal["message.updated"] = "message.updated"
    properties: MessageUpdatedEventProperties


class PartUpdatedEventProperties(OpenCodeBaseModel):
    """Properties for part updated event."""

    part: Part
    delta: str | None = None


class PartUpdatedEvent(OpenCodeBaseModel):
    """Part updated event."""

    type: Literal["message.part.updated"] = "message.part.updated"
    properties: PartUpdatedEventProperties


class PermissionRequestProperties(OpenCodeBaseModel):
    """Properties for permission request event."""

    session_id: str
    permission_id: str
    tool_name: str
    args_preview: str
    message: str


class PermissionRequestEvent(OpenCodeBaseModel):
    """Permission request event - sent when a tool needs user confirmation."""

    type: Literal["permission.request"] = "permission.request"
    properties: PermissionRequestProperties


class PermissionResolvedProperties(OpenCodeBaseModel):
    """Properties for permission resolved event."""

    session_id: str
    permission_id: str
    response: str  # "once" | "always" | "reject"


class PermissionResolvedEvent(OpenCodeBaseModel):
    """Permission resolved event - sent when a permission request is answered."""

    type: Literal["permission.resolved"] = "permission.resolved"
    properties: PermissionResolvedProperties


Event = (
    ServerConnectedEvent
    | SessionCreatedEvent
    | SessionUpdatedEvent
    | SessionDeletedEvent
    | SessionStatusEvent
    | SessionIdleEvent
    | MessageUpdatedEvent
    | PartUpdatedEvent
    | PermissionRequestEvent
    | PermissionResolvedEvent
)
