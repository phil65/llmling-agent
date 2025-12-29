"""Message part models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class TimeStart(OpenCodeBaseModel):
    """Time with only start (milliseconds).

    Used by: ToolStateRunning
    """

    start: int


class TimeStartEnd(OpenCodeBaseModel):
    """Time with start and end, both required (milliseconds).

    Used by: ToolStateError
    """

    start: int
    end: int


class TimeStartEndOptional(OpenCodeBaseModel):
    """Time with start required and end optional (milliseconds).

    Used by: TextPart
    """

    start: int
    end: int | None = None


class TimeStartEndCompacted(OpenCodeBaseModel):
    """Time with start, end required, and optional compacted (milliseconds).

    Used by: ToolStateCompleted
    """

    start: int
    end: int
    compacted: int | None = None


class TextPart(OpenCodeBaseModel):
    """Text content part."""

    id: str
    type: Literal["text"] = "text"
    message_id: str
    session_id: str
    text: str
    synthetic: bool | None = None
    ignored: bool | None = None
    time: TimeStartEndOptional | None = None
    metadata: dict[str, Any] | None = None


class ToolStatePending(OpenCodeBaseModel):
    """Pending tool state."""

    status: Literal["pending"] = "pending"
    input: dict[str, Any] = Field(default_factory=dict)
    raw: str = ""


class ToolStateRunning(OpenCodeBaseModel):
    """Running tool state."""

    status: Literal["running"] = "running"
    time: TimeStart
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None
    title: str | None = None


class ToolStateCompleted(OpenCodeBaseModel):
    """Completed tool state."""

    status: Literal["completed"] = "completed"
    input: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    time: TimeStartEndCompacted
    attachments: list[Any] | None = None


class ToolStateError(OpenCodeBaseModel):
    """Error tool state."""

    status: Literal["error"] = "error"
    input: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] | None = None
    time: TimeStartEnd


ToolState = ToolStatePending | ToolStateRunning | ToolStateCompleted | ToolStateError


class ToolPart(OpenCodeBaseModel):
    """Tool call part."""

    id: str
    type: Literal["tool"] = "tool"
    message_id: str
    session_id: str
    call_id: str
    tool: str
    state: ToolState
    metadata: dict[str, Any] | None = None


class FilePartSourceText(OpenCodeBaseModel):
    """File part source text."""

    value: str
    start: int
    end: int


class FileSource(OpenCodeBaseModel):
    """File source."""

    text: FilePartSourceText
    type: Literal["file"] = "file"
    path: str


class FilePart(OpenCodeBaseModel):
    """File content part."""

    id: str
    type: Literal["file"] = "file"
    message_id: str
    session_id: str
    mime: str
    filename: str | None = None
    url: str
    source: FileSource | None = None


class StepStartPart(OpenCodeBaseModel):
    """Step start marker."""

    id: str
    type: Literal["step-start"] = "step-start"
    message_id: str
    session_id: str
    snapshot: str | None = None


class TokenCache(OpenCodeBaseModel):
    """Token cache information."""

    read: int = 0
    write: int = 0


class StepFinishTokens(OpenCodeBaseModel):
    """Token usage for step finish."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache: TokenCache = Field(default_factory=TokenCache)


class StepFinishPart(OpenCodeBaseModel):
    """Step finish marker."""

    id: str
    type: Literal["step-finish"] = "step-finish"
    message_id: str
    session_id: str
    reason: str = "stop"
    snapshot: str | None = None
    cost: int = 0
    tokens: StepFinishTokens = Field(default_factory=StepFinishTokens)


Part = TextPart | ToolPart | FilePart | StepStartPart | StepFinishPart
