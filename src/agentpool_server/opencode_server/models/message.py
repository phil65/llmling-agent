"""Message related models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agentpool.utils import identifiers as identifier
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import (  # noqa: TC001
    FileDiff,
    ModelRef,
    TimeCreated,
    Tokens,
)
from agentpool_server.opencode_server.models.parts import (  # noqa: TC001
    AgentPart,
    AgentPartSource,
    APIErrorInfo,
    FilePart,
    FilePartSource,
    Part,
    RetryPart,
    StepFinishPart,
    StepStartPart,
    SubtaskPart,
    TextPart,
    ToolPart,
    ToolState,
)


class MessageSummary(OpenCodeBaseModel):
    """Summary information for a message."""

    title: str | None = None
    body: str | None = None
    diffs: list[FileDiff] = Field(default_factory=list)


class MessagePath(OpenCodeBaseModel):
    """Path context for a message."""

    cwd: str
    root: str


class MessageTime(OpenCodeBaseModel):
    """Time information for a message (milliseconds)."""

    created: int
    completed: int | None = None


class UserMessage(OpenCodeBaseModel):
    """User message."""

    id: str
    role: Literal["user"] = "user"
    session_id: str
    time: TimeCreated
    agent: str = "default"
    model: ModelRef | None = None
    summary: MessageSummary | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None
    variant: str | None = None


# --- Assistant message error types ---
# These match the NamedError pattern from upstream OpenCode:
# Each error is { name: Literal["..."], data: { ... } }


class ProviderAuthErrorData(OpenCodeBaseModel):
    """Data for provider authentication errors."""

    provider_id: str
    message: str


class ProviderAuthError(OpenCodeBaseModel):
    """Provider authentication error."""

    name: Literal["ProviderAuthError"] = Field(default="ProviderAuthError", init=False)
    data: ProviderAuthErrorData


class UnknownErrorData(OpenCodeBaseModel):
    """Data for unknown errors."""

    message: str


class UnknownError(OpenCodeBaseModel):
    """Unknown error."""

    name: Literal["UnknownError"] = Field(default="UnknownError", init=False)
    data: UnknownErrorData


class MessageOutputLengthErrorData(OpenCodeBaseModel):
    """Data for output length errors (empty)."""


class MessageOutputLengthError(OpenCodeBaseModel):
    """Message output length exceeded error."""

    name: Literal["MessageOutputLengthError"] = Field(
        default="MessageOutputLengthError", init=False
    )
    data: MessageOutputLengthErrorData = Field(default_factory=MessageOutputLengthErrorData)


class MessageAbortedErrorData(OpenCodeBaseModel):
    """Data for aborted message errors."""

    message: str


class MessageAbortedError(OpenCodeBaseModel):
    """Message was aborted."""

    name: Literal["MessageAbortedError"] = Field(default="MessageAbortedError", init=False)
    data: MessageAbortedErrorData


class APIErrorData(OpenCodeBaseModel):
    """Data for API errors."""

    message: str
    status_code: int | None = None
    is_retryable: bool = False
    response_headers: dict[str, str] | None = None
    response_body: str | None = None
    metadata: dict[str, str] | None = None


class APIError(OpenCodeBaseModel):
    """API error."""

    name: Literal["APIError"] = Field(default="APIError", init=False)
    data: APIErrorData


MessageError = (
    ProviderAuthError | UnknownError | MessageOutputLengthError | MessageAbortedError | APIError
)


class AssistantMessage(OpenCodeBaseModel):
    """Assistant message."""

    id: str
    role: Literal["assistant"] = "assistant"
    session_id: str
    parent_id: str  # Required - links to user message
    model_id: str
    provider_id: str
    mode: str = "default"
    agent: str = "default"
    path: MessagePath
    time: MessageTime
    tokens: Tokens = Field(default_factory=Tokens)
    cost: float = 0.0
    error: MessageError | None = None
    summary: bool | None = None
    finish: str | None = None


class MessageWithParts(OpenCodeBaseModel):
    """Message with its parts."""

    info: UserMessage | AssistantMessage
    parts: list[Part] = Field(default_factory=list)

    def update_part(self, updated: Part) -> None:
        """Replace a part in the assistant message's parts list by ID."""
        for i, p in enumerate(self.parts):
            if isinstance(p, type(updated)) and p.id == updated.id:
                self.parts[i] = updated
                break

    def add_text_part(self, text: str, **kwargs: Any) -> TextPart:
        """Create and append a text part."""
        part = TextPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            text=text,
            **kwargs,
        )
        self.parts.append(part)
        return part

    def add_file_part(
        self,
        mime: str,
        url: str,
        filename: str | None = None,
        source: FilePartSource | None = None,
    ) -> FilePart:
        """Create and append a file part."""
        part = FilePart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            mime=mime,
            url=url,
            filename=filename,
            source=source,
        )
        self.parts.append(part)
        return part

    def add_agent_part(
        self,
        name: str,
        source: AgentPartSource | None = None,
    ) -> AgentPart:
        """Create and append an agent mention part."""
        part = AgentPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            name=name,
            source=source,
        )
        self.parts.append(part)
        return part

    def add_subtask_part(
        self,
        prompt: str,
        description: str,
        agent: str,
        model: ModelRef | None = None,
    ) -> SubtaskPart:
        """Create and append a subtask part."""
        part = SubtaskPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            prompt=prompt,
            description=description,
            agent=agent,
            model=model,
        )
        self.parts.append(part)
        return part

    def add_step_start_part(self, snapshot: str | None = None) -> StepStartPart:
        """Create and append a step start marker."""
        part = StepStartPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            snapshot=snapshot,
        )
        self.parts.append(part)
        return part

    def add_step_finish_part(
        self,
        reason: str = "stop",
        cost: float = 0.0,
        tokens: Tokens | None = None,
        snapshot: str | None = None,
    ) -> StepFinishPart:
        """Create and append a step finish marker."""
        part = StepFinishPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            reason=reason,
            cost=cost,
            tokens=tokens or Tokens(),
            snapshot=snapshot,
        )
        self.parts.append(part)
        return part

    def add_tool_part(
        self,
        tool: str,
        call_id: str,
        state: ToolState,
    ) -> ToolPart:
        """Create and append a tool call part."""
        part = ToolPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            tool=tool,
            call_id=call_id,
            state=state,
        )
        self.parts.append(part)
        return part

    def add_retry_part(
        self,
        attempt: int,
        message: str,
        created: int,
        is_retryable: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> RetryPart:
        """Create and append a retry part."""
        part = RetryPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            attempt=attempt,
            error=APIErrorInfo(
                message=message,
                is_retryable=is_retryable,
                metadata=metadata,
            ),
            time=TimeCreated(created=created),
        )
        self.parts.append(part)
        return part


class TextPartInput(OpenCodeBaseModel):
    """Text part for input."""

    type: Literal["text"] = Field(default="text", init=False)
    text: str


class FilePartInput(OpenCodeBaseModel):
    """File part for input (image, document, etc.)."""

    type: Literal["file"] = Field(default="file", init=False)
    mime: str
    filename: str | None = None
    url: str  # Can be data: URI or file path
    source: FilePartSource | None = None


class AgentPartInput(OpenCodeBaseModel):
    """Agent mention part for input - references a sub-agent to delegate to.

    When a user types @agent-name in the prompt, this part is created.
    """

    type: Literal["agent"] = Field(default="agent", init=False)
    name: str
    """Name of the agent to delegate to."""
    source: AgentPartSource | None = None
    """Source location in the original prompt text."""


class SubtaskPartInput(OpenCodeBaseModel):
    """Subtask part for input - spawns a subtask to another agent."""

    type: Literal["subtask"] = Field(default="subtask", init=False)
    prompt: str
    """The prompt for the subtask."""
    description: str
    """Description of what the subtask does."""
    agent: str
    """The agent to handle this subtask."""
    model: ModelRef | None = None
    """Optional model to use for the subtask."""


PartInput = TextPartInput | FilePartInput | AgentPartInput | SubtaskPartInput


class MessageRequest(OpenCodeBaseModel):
    """Request body for sending a message."""

    parts: list[PartInput]
    message_id: str | None = None
    model: ModelRef | None = None
    agent: str | None = None
    no_reply: bool | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None
    variant: str | None = None
    """Reasoning/thinking variant for this message.

    Maps to the model's variants (e.g., 'low', 'medium', 'high', 'max').
    When set, the agent will use this thinking effort level for the response.
    """


class ShellRequest(OpenCodeBaseModel):
    """Request body for running a shell command."""

    agent: str
    command: str
    model: ModelRef | None = None


class CommandRequest(OpenCodeBaseModel):
    """Request body for executing a slash command."""

    command: str
    arguments: str | None = None
    agent: str | None = None
    model: str | None = None  # Format: "providerID/modelID"
    message_id: str | None = None


# Type unions

MessageInfo = UserMessage | AssistantMessage
