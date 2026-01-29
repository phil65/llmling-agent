"""Message part models."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field

from agentpool.utils.time_utils import now_ms
from agentpool_server.opencode_server.models.agent import AgentModel  # noqa: TC001
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeCreated  # noqa: TC001


class TimeStart(OpenCodeBaseModel):
    """Time with only start (milliseconds).

    Used by: ToolStateRunning
    """

    start: int

    @classmethod
    def now(cls) -> Self:
        return cls(start=now_ms())


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


class AgentPartSource(OpenCodeBaseModel):
    """Agent part source location in original text."""

    value: str
    start: int
    end: int


class AgentPart(OpenCodeBaseModel):
    """Agent mention part - references a sub-agent to delegate to.

    When a user types @agent-name in the prompt, this part is created.
    The server should inject a synthetic instruction to call the task tool
    with the specified agent.
    """

    id: str
    type: Literal["agent"] = "agent"
    message_id: str
    session_id: str
    name: str
    """Name of the agent to delegate to."""
    source: AgentPartSource | None = None
    """Source location in the original prompt text."""


class SnapshotPart(OpenCodeBaseModel):
    """File system snapshot reference."""

    id: str
    type: Literal["snapshot"] = "snapshot"
    message_id: str
    session_id: str
    snapshot: str
    """Snapshot identifier."""


class PatchPart(OpenCodeBaseModel):
    """Diff/patch content part."""

    id: str
    type: Literal["patch"] = "patch"
    message_id: str
    session_id: str
    hash: str
    """Hash of the patch."""
    files: list[str] = Field(default_factory=list)
    """List of files affected by this patch."""


class ReasoningPart(OpenCodeBaseModel):
    """Extended thinking/reasoning content part.

    Used for models that support extended thinking (e.g., Claude with thinking tokens).
    """

    id: str
    type: Literal["reasoning"] = "reasoning"
    message_id: str
    session_id: str
    text: str
    """The reasoning/thinking content."""
    metadata: dict[str, Any] | None = None
    time: TimeStartEndOptional | None = None


class CompactionPart(OpenCodeBaseModel):
    """Marks where conversation was compacted/summarized."""

    id: str
    type: Literal["compaction"] = "compaction"
    message_id: str
    session_id: str
    auto: bool = False
    """Whether this was an automatic compaction."""


class SubtaskPart(OpenCodeBaseModel):
    """References a spawned subtask."""

    id: str
    type: Literal["subtask"] = "subtask"
    message_id: str
    session_id: str
    prompt: str
    """The prompt for the subtask."""
    description: str
    """Description of what the subtask does."""
    agent: str
    """The agent handling this subtask."""
    command: str | None = None
    """Optional command associated with the subtask."""
    model: AgentModel | None = None
    """The model used for the subtask."""


class APIErrorInfo(OpenCodeBaseModel):
    """API error information for retry parts."""

    message: str
    status_code: int | None = None
    is_retryable: bool = False
    response_headers: dict[str, str] | None = None
    response_body: str | None = None
    metadata: dict[str, str] | None = None


class RetryPart(OpenCodeBaseModel):
    """Marks a retry of a failed operation."""

    id: str
    type: Literal["retry"] = "retry"
    message_id: str
    session_id: str
    attempt: int
    """Which retry attempt this is."""
    error: APIErrorInfo
    """Error information from the failed attempt."""
    time: TimeCreated


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
    cost: float = 0.0
    tokens: StepFinishTokens = Field(default_factory=StepFinishTokens)


Part = (
    TextPart
    | ToolPart
    | FilePart
    | AgentPart
    | SnapshotPart
    | PatchPart
    | ReasoningPart
    | CompactionPart
    | SubtaskPart
    | RetryPart
    | StepStartPart
    | StepFinishPart
)
