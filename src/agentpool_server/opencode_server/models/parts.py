"""Message part models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeStartEnd  # noqa: TC001


class TextPart(OpenCodeBaseModel):
    """Text content part."""

    id: str
    type: Literal["text"] = "text"
    message_id: str = Field(alias="messageID")
    session_id: str = Field(alias="sessionID")
    text: str
    synthetic: bool | None = None
    time: TimeStartEnd | None = None


class ToolStatePending(OpenCodeBaseModel):
    """Pending tool state."""

    status: Literal["pending"] = "pending"


class ToolStateRunning(OpenCodeBaseModel):
    """Running tool state."""

    status: Literal["running"] = "running"
    input: dict[str, Any] = Field(default_factory=dict)
    time: TimeStartEnd | None = None


class ToolStateCompleted(OpenCodeBaseModel):
    """Completed tool state."""

    status: Literal["completed"] = "completed"
    input: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    time: TimeStartEnd | None = None


class ToolStateError(OpenCodeBaseModel):
    """Error tool state."""

    status: Literal["error"] = "error"
    input: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    time: TimeStartEnd | None = None


ToolState = ToolStatePending | ToolStateRunning | ToolStateCompleted | ToolStateError


class ToolPart(OpenCodeBaseModel):
    """Tool call part."""

    id: str
    type: Literal["tool"] = "tool"
    message_id: str = Field(alias="messageID")
    session_id: str = Field(alias="sessionID")
    call_id: str = Field(alias="callID")
    tool: str
    state: ToolState


class FilePart(OpenCodeBaseModel):
    """File content part."""

    id: str
    type: Literal["file"] = "file"
    message_id: str = Field(alias="messageID")
    session_id: str = Field(alias="sessionID")
    path: str
    content: str | None = None


class StepStartPart(OpenCodeBaseModel):
    """Step start marker."""

    id: str
    type: Literal["step-start"] = "step-start"
    message_id: str = Field(alias="messageID")
    session_id: str = Field(alias="sessionID")


class StepFinishPart(OpenCodeBaseModel):
    """Step finish marker."""

    id: str
    type: Literal["step-finish"] = "step-finish"
    message_id: str = Field(alias="messageID")
    session_id: str = Field(alias="sessionID")


Part = TextPart | ToolPart | FilePart | StepStartPart | StepFinishPart
