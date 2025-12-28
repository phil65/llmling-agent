"""SSE event models."""

from __future__ import annotations

from typing import Literal

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.message import (  # noqa: TC001
    AssistantMessage,
    UserMessage,
)
from agentpool_server.opencode_server.models.parts import Part  # noqa: TC001
from agentpool_server.opencode_server.models.session import Session  # noqa: TC001


class ServerConnectedEvent(OpenCodeBaseModel):
    """Server connected event."""

    type: Literal["server.connected"] = "server.connected"


class SessionCreatedEvent(OpenCodeBaseModel):
    """Session created event."""

    type: Literal["session.created"] = "session.created"
    properties: Session


class SessionUpdatedEvent(OpenCodeBaseModel):
    """Session updated event."""

    type: Literal["session.updated"] = "session.updated"
    properties: Session


class SessionDeletedEvent(OpenCodeBaseModel):
    """Session deleted event."""

    type: Literal["session.deleted"] = "session.deleted"
    properties: dict[str, str]  # {id: session_id}


class MessageCreatedEvent(OpenCodeBaseModel):
    """Message created event."""

    type: Literal["message.created"] = "message.created"
    properties: UserMessage | AssistantMessage


class MessageUpdatedEvent(OpenCodeBaseModel):
    """Message updated event."""

    type: Literal["message.updated"] = "message.updated"
    properties: UserMessage | AssistantMessage


class PartUpdatedEvent(OpenCodeBaseModel):
    """Part updated event."""

    type: Literal["part.updated"] = "part.updated"
    properties: Part


Event = (
    ServerConnectedEvent
    | SessionCreatedEvent
    | SessionUpdatedEvent
    | SessionDeletedEvent
    | MessageCreatedEvent
    | MessageUpdatedEvent
    | PartUpdatedEvent
)
