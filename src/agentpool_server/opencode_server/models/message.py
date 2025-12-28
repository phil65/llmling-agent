"""Message related models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeCreated  # noqa: TC001
from agentpool_server.opencode_server.models.parts import Part  # noqa: TC001


class MessagePath(OpenCodeBaseModel):
    """Path context for a message."""

    cwd: str
    root: str


class MessageTime(OpenCodeBaseModel):
    """Time information for a message."""

    created: float
    completed: float | None = None


class TokensCache(OpenCodeBaseModel):
    """Token cache information."""

    read: float = 0
    write: float = 0


class Tokens(OpenCodeBaseModel):
    """Token usage information."""

    cache: TokensCache = Field(default_factory=TokensCache)
    input: float = 0
    output: float = 0
    reasoning: float = 0


class UserMessage(OpenCodeBaseModel):
    """User message."""

    id: str
    role: Literal["user"] = "user"
    session_id: str = Field(alias="sessionID")
    time: TimeCreated


class AssistantMessage(OpenCodeBaseModel):
    """Assistant message."""

    id: str
    role: Literal["assistant"] = "assistant"
    session_id: str = Field(alias="sessionID")
    model_id: str = Field(alias="modelID")
    provider_id: str = Field(alias="providerID")
    mode: str = "default"
    path: MessagePath
    system: list[str] = Field(default_factory=list)
    time: MessageTime
    tokens: Tokens = Field(default_factory=Tokens)
    cost: float = 0.0
    error: dict[str, Any] | None = None
    summary: bool | None = None


class MessageWithParts(OpenCodeBaseModel):
    """Message with its parts."""

    info: UserMessage | AssistantMessage
    parts: list[Part] = Field(default_factory=list)


# Request models


class TextPartInput(OpenCodeBaseModel):
    """Text part for input."""

    type: Literal["text"] = "text"
    text: str


class FilePartInput(OpenCodeBaseModel):
    """File part for input."""

    type: Literal["file"] = "file"
    path: str
    content: str | None = None


PartInput = TextPartInput | FilePartInput


class MessageRequest(OpenCodeBaseModel):
    """Request body for sending a message."""

    parts: list[PartInput]
    provider_id: str = Field(alias="providerID")
    model_id: str = Field(alias="modelID")
    message_id: str | None = Field(default=None, alias="messageID")
    mode: str | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None
