"""Session update schema definitions."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Annotated, Any, Literal

from pydantic import Field

from acp.base import AnnotatedObject
from acp.schema.agent_plan import PlanEntry  # noqa: TC001
from acp.schema.content_blocks import ContentBlock  # noqa: TC001
from acp.schema.slash_commands import AvailableCommand  # noqa: TC001
from acp.schema.tool_call import (  # noqa: TC001
    ToolCallContent,
    ToolCallKind,
    ToolCallLocation,
)


ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]


class UserMessageChunk(AnnotatedObject):
    """A chunk of the user's message being streamed."""

    session_update: Literal["user_message_chunk"] = Field(
        default="user_message_chunk", init=False
    )
    """User message chunk."""

    content: ContentBlock
    """A single item of content"""


class AgentMessageChunk(AnnotatedObject):
    """A chunk of the agent's response being streamed."""

    session_update: Literal["agent_message_chunk"] = Field(
        default="agent_message_chunk", init=False
    )
    """Agent message chunk."""

    content: ContentBlock
    """A single item of content"""


class AgentThoughtChunk(AnnotatedObject):
    """A chunk of the agent's internal reasoning being streamed."""

    session_update: Literal["agent_thought_chunk"] = Field(
        default="agent_thought_chunk", init=False
    )
    """Agent thought chunk."""

    content: ContentBlock
    """A single item of content"""


class ToolCallProgress(AnnotatedObject):
    """Update on the status or results of a tool call."""

    session_update: Literal["tool_call_update"] = Field(
        default="tool_call_update", init=False
    )
    """Tool call update."""

    content: Sequence[ToolCallContent] | None = None
    """Replace the content collection."""

    kind: ToolCallKind | None = None
    """Update the tool kind."""

    locations: Sequence[ToolCallLocation] | None = None
    """Replace the locations collection."""

    raw_input: Any | None = None
    """Update the raw input."""

    raw_output: Any | None = None
    """Update the raw output."""

    status: ToolCallStatus | None = None
    """Update the execution status."""

    title: str | None = None
    """Update the human-readable title."""

    tool_call_id: str
    """The ID of the tool call being updated."""


class CurrentModeUpdate(AnnotatedObject):
    """The current mode of the session has changed.

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """

    current_mode_id: str
    """The ID of the current mode"""

    session_update: Literal["current_mode_update"] = Field(
        default="current_mode_update", init=False
    )


class AgentPlanUpdate(AnnotatedObject):
    """The agent's execution plan for complex tasks.

    See protocol docs: [Agent Plan](https://agentclientprotocol.com/protocol/agent-plan).
    """

    session_update: Literal["plan"] = Field(default="plan", init=False)

    entries: Sequence[PlanEntry]
    """The list of tasks to be accomplished.

    When updating a plan, the agent must send a complete list of all entries
    with their current status. The client replaces the entire plan with each update."""


class AvailableCommandsUpdate(AnnotatedObject):
    """Available commands are ready or have changed."""

    session_update: Literal["available_commands_update"] = Field(
        default="available_commands_update", init=False
    )
    """Available commands are ready or have changed."""

    available_commands: Sequence[AvailableCommand]
    """Commands the agent can execute"""


class ToolCallStart(AnnotatedObject):
    """Notification that a new tool call has been initiated."""

    session_update: Literal["tool_call"] = Field(default="tool_call", init=False)
    """Notification that a new tool call has been initiated."""

    content: Sequence[ToolCallContent] | None = None
    """Content produced by the tool call."""

    kind: ToolCallKind | None = None
    """The category of tool being invoked.

    Helps clients choose appropriate icons and UI treatment.
    """

    locations: Sequence[ToolCallLocation] | None = None
    """File locations affected by this tool call.

    Enables "follow-along" features in clients.
    """

    raw_input: Any | None = None
    """Raw input parameters sent to the tool."""

    raw_output: Any | None = None
    """Raw output returned by the tool."""

    status: ToolCallStatus | None = None
    """Current execution status of the tool call."""

    title: str
    """Human-readable title describing what the tool is doing."""

    tool_call_id: str
    """Unique identifier for this tool call within the session."""


SessionUpdate = Annotated[
    (
        UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AvailableCommandsUpdate
        | AgentPlanUpdate
        | CurrentModeUpdate
    ),
    Field(discriminator="session_update"),
]
