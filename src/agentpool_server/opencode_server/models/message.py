"""Message related models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from agentpool_server.opencode_server.models.agent import AgentModel  # noqa: TC001
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeCreated  # noqa: TC001
from agentpool_server.opencode_server.models.parts import (  # noqa: TC001
    AgentPartSource,
    Part,
)


if TYPE_CHECKING:
    from agentpool_server.opencode_server.models.parts import ReasoningPart, TextPart, ToolPart


class MessageSummary(OpenCodeBaseModel):
    """Summary information for a message."""

    title: str | None = None
    body: str | None = None
    diffs: list[Any] = Field(default_factory=list)


class MessagePath(OpenCodeBaseModel):
    """Path context for a message."""

    cwd: str
    root: str


class MessageTime(OpenCodeBaseModel):
    """Time information for a message (milliseconds)."""

    created: int
    completed: int | None = None


class TokensCache(OpenCodeBaseModel):
    """Token cache information."""

    read: int = 0
    write: int = 0


class Tokens(OpenCodeBaseModel):
    """Token usage information."""

    cache: TokensCache = Field(default_factory=TokensCache)
    input: int = 0
    output: int = 0
    reasoning: int = 0


class UserMessageModel(OpenCodeBaseModel):
    """Model info for user message."""

    provider_id: str
    model_id: str


class UserMessage(OpenCodeBaseModel):
    """User message."""

    id: str
    role: Literal["user"] = "user"
    session_id: str
    time: TimeCreated
    agent: str = "default"
    model: UserMessageModel | None = None
    summary: MessageSummary | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None
    variant: str | None = None


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
    error: dict[str, Any] | None = None
    summary: bool | None = None
    finish: str | None = None


class MessageWithParts(OpenCodeBaseModel):
    """Message with its parts."""

    info: UserMessage | AssistantMessage
    parts: list[Part] = Field(default_factory=list)

    def update_part(
        self,
        part_id: str,
        updated: TextPart | ToolPart | ReasoningPart,
        part_type: type[TextPart | ToolPart | ReasoningPart],
    ) -> None:
        """Replace a part in the assistant message's parts list by ID."""
        for i, p in enumerate(self.parts):
            if isinstance(p, part_type) and p.id == part_id:
                self.parts[i] = updated
                break


class TextPartInput(OpenCodeBaseModel):
    """Text part for input."""

    type: Literal["text"] = "text"
    text: str


class FilePartSourceText(OpenCodeBaseModel):
    """Source text info for file part."""

    value: str
    start: int
    end: int


class FilePartSource(OpenCodeBaseModel):
    """Source info for file part."""

    text: FilePartSourceText | None = None
    type: str | None = None
    path: str | None = None

    @classmethod
    def create(
        cls,
        value: str,
        start: int,
        end: int,
        typ: str | None = None,
        path: str | None = None,
    ) -> FilePartSource:
        return cls(text=FilePartSourceText(value=value, start=start, end=end), type=typ, path=path)


class FilePartInput(OpenCodeBaseModel):
    """File part for input (image, document, etc.)."""

    type: Literal["file"] = "file"
    mime: str
    filename: str | None = None
    url: str  # Can be data: URI or file path
    source: FilePartSource | None = None


class AgentPartInput(OpenCodeBaseModel):
    """Agent mention part for input - references a sub-agent to delegate to.

    When a user types @agent-name in the prompt, this part is created.
    """

    type: Literal["agent"] = "agent"
    name: str
    """Name of the agent to delegate to."""
    source: AgentPartSource | None = None
    """Source location in the original prompt text."""


class SubtaskPartInput(OpenCodeBaseModel):
    """Subtask part for input - spawns a subtask to another agent."""

    type: Literal["subtask"] = "subtask"
    prompt: str
    """The prompt for the subtask."""
    description: str
    """Description of what the subtask does."""
    agent: str
    """The agent to handle this subtask."""
    model: AgentModel | None = None
    """Optional model to use for the subtask."""


PartInput = TextPartInput | FilePartInput | AgentPartInput | SubtaskPartInput


class MessageModelInfo(OpenCodeBaseModel):
    """Model info in message request."""

    provider_id: str
    model_id: str


class MessageRequest(OpenCodeBaseModel):
    """Request body for sending a message."""

    parts: list[PartInput]
    message_id: str | None = None
    model: MessageModelInfo | None = None
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
    model: MessageModelInfo | None = None


class CommandRequest(OpenCodeBaseModel):
    """Request body for executing a slash command."""

    command: str
    arguments: str | None = None
    agent: str | None = None
    model: str | None = None  # Format: "providerID/modelID"
    message_id: str | None = None


# Type unions

MessageInfo = UserMessage | AssistantMessage
