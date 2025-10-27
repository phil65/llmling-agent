"""Tool call schema definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import Field

from acp.base import Schema
from acp.schema.content_blocks import ContentBlock


ToolCallKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
]
ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
PermissionKind = Literal["allow_once", "allow_always", "reject_once", "reject_always"]


class ToolCall(Schema):
    """Details about the tool call requiring permission."""

    field_meta: Any | None = None
    """Extension point for implementations."""

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


class FileEditToolCallContent(Schema):
    """File modification shown as a diff."""

    type: Literal["diff"] = "diff"
    """File modification shown as a diff."""

    field_meta: Any | None = None
    """Extension point for implementations."""

    new_text: str
    """The new content after modification."""

    old_text: str | None
    """The original content (None for new files)."""

    path: str
    """The file path being modified."""


class TerminalToolCallContent(Schema):
    """Embed a terminal created with `terminal/create` by its id.

    The terminal must be added before calling `terminal/release`.
    See protocol docs: [Terminal](https://agentclientprotocol.com/protocol/terminal)
    """

    type: Literal["terminal"] = "terminal"
    """Terminal tool call content."""

    terminal_id: str
    """The ID of the terminal being embedded."""


class ContentToolCallContent(Schema):
    """Standard content block (text, images, resources)."""

    type: Literal["content"] = "content"
    """Standard content block (text, images, resources)."""

    content: ContentBlock
    """The actual content block."""


class ToolCallLocation(Schema):
    """A file location being accessed or modified by a tool.

    Enables clients to implement "follow-along" features that track
    which files the agent is working with in real-time.
    See protocol docs: [Following the Agent](https://agentclientprotocol.com/protocol/tool-calls#following-the-agent)
    """

    field_meta: Any | None = None
    """Extension point for implementations."""

    line: int | None = Field(default=None, ge=0)
    """Optional line number within the file."""

    path: Annotated[str, Field(description="The file path being accessed or modified.")]


class DeniedOutcome(Schema):
    """The prompt turn was cancelled before the user responded.

    When a client sends a `session/cancel` notification to cancel an ongoing
    prompt turn, it MUST respond to all pending `session/request_permission`
    requests with this `Cancelled` outcome.
    See protocol docs: [Cancellation](https://agentclientprotocol.com/protocol/prompt-turn#cancellation)
    """

    outcome: Literal["cancelled"] = "cancelled"


class AllowedOutcome(Schema):
    """The user selected one of the provided options."""

    option_id: str
    """The ID of the option the user selected."""

    outcome: Literal["selected"] = "selected"


class PermissionOption(Schema):
    """An option presented to the user when requesting permission."""

    field_meta: Any | None = None
    """Extension point for implementations."""

    kind: PermissionKind
    """Hint about the nature of this permission option."""

    name: str
    """Human-readable label to display to the user."""

    option_id: str
    """Unique identifier for this permission option."""


ToolCallContent = (
    ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent
)
