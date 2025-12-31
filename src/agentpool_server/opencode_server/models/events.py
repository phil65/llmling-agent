"""SSE event models."""

from __future__ import annotations

from typing import Literal, Self

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

    @classmethod
    def create(cls, session_id: str) -> Self:
        return cls(properties=SessionDeletedProperties(session_id=session_id))


class SessionStatusProperties(OpenCodeBaseModel):
    """Properties for session status event."""

    session_id: str
    status: SessionStatus


class SessionStatusEvent(OpenCodeBaseModel):
    """Session status event."""

    type: Literal["session.status"] = "session.status"
    properties: SessionStatusProperties

    @classmethod
    def create(cls, session_id: str, status: SessionStatus) -> Self:
        return cls(properties=SessionStatusProperties(session_id=session_id, status=status))


class SessionIdleProperties(OpenCodeBaseModel):
    """Properties for session idle event (deprecated but still used by TUI)."""

    session_id: str


class SessionIdleEvent(OpenCodeBaseModel):
    """Session idle event (deprecated but still used by TUI run command)."""

    type: Literal["session.idle"] = "session.idle"
    properties: SessionIdleProperties

    @classmethod
    def create(cls, session_id: str) -> Self:
        return cls(properties=SessionIdleProperties(session_id=session_id))


class MessageUpdatedEventProperties(OpenCodeBaseModel):
    """Properties for message updated event."""

    info: UserMessage | AssistantMessage


class MessageUpdatedEvent(OpenCodeBaseModel):
    """Message updated event."""

    type: Literal["message.updated"] = "message.updated"
    properties: MessageUpdatedEventProperties

    @classmethod
    def create(cls, message: UserMessage | AssistantMessage) -> Self:
        return cls(properties=MessageUpdatedEventProperties(info=message))


class PartUpdatedEventProperties(OpenCodeBaseModel):
    """Properties for part updated event."""

    part: Part
    delta: str | None = None


class PartUpdatedEvent(OpenCodeBaseModel):
    """Part updated event."""

    type: Literal["message.part.updated"] = "message.part.updated"
    properties: PartUpdatedEventProperties

    @classmethod
    def create(cls, part: Part, delta: str | None = None) -> Self:
        return cls(properties=PartUpdatedEventProperties(part=part, delta=delta))


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

    @classmethod
    def create(
        cls,
        session_id: str,
        permission_id: str,
        tool_name: str,
        args_preview: str,
        message: str,
    ) -> Self:
        props = PermissionRequestProperties(
            session_id=session_id,
            permission_id=permission_id,
            tool_name=tool_name,
            args_preview=args_preview,
            message=message,
        )
        return cls(properties=props)


class PermissionResolvedProperties(OpenCodeBaseModel):
    """Properties for permission resolved event."""

    session_id: str
    permission_id: str
    response: str  # "once" | "always" | "reject"


class PermissionResolvedEvent(OpenCodeBaseModel):
    """Permission resolved event - sent when a permission request is answered."""

    type: Literal["permission.resolved"] = "permission.resolved"
    properties: PermissionResolvedProperties

    @classmethod
    def create(
        cls,
        session_id: str,
        permission_id: str,
        response: str,
    ) -> Self:
        props = PermissionResolvedProperties(
            session_id=session_id,
            permission_id=permission_id,
            response=response,
        )
        return cls(properties=props)


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
