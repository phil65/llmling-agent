"""Tool call schema definitions."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Any, Literal

from pydantic import Field

from acp.schema.base import AnnotatedObject, Schema
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


class ToolCall(AnnotatedObject):
    """Details about the tool call requiring permission."""

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


class FileEditToolCallContent(AnnotatedObject):
    """File modification shown as a diff."""

    type: Literal["diff"] = Field(default="diff", init=False)
    """File modification shown as a diff."""

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

    type: Literal["terminal"] = Field(default="terminal", init=False)
    """Terminal tool call content."""

    terminal_id: str
    """The ID of the terminal being embedded."""


class ContentToolCallContent[TContentBlock: ContentBlock = ContentBlock](Schema):
    """Standard content block (text, images, resources)."""

    type: Literal["content"] = Field(default="content", init=False)
    """Standard content block (text, images, resources)."""

    content: TContentBlock
    """The actual content block."""


class ToolCallLocation(AnnotatedObject):
    """A file location being accessed or modified by a tool.

    Enables clients to implement "follow-along" features that track
    which files the agent is working with in real-time.
    See protocol docs: [Following the Agent](https://agentclientprotocol.com/protocol/tool-calls#following-the-agent)
    """

    line: int | None = Field(default=None, ge=0)
    """Optional line number within the file."""

    path: str
    """The file path being accessed or modified."""


class DeniedOutcome(Schema):
    """The prompt turn was cancelled before the user responded.

    When a client sends a `session/cancel` notification to cancel an ongoing
    prompt turn, it MUST respond to all pending `session/request_permission`
    requests with this `Cancelled` outcome.
    See protocol docs: [Cancellation](https://agentclientprotocol.com/protocol/prompt-turn#cancellation)
    """

    outcome: Literal["cancelled"] = Field(default="cancelled", init=False)


class AllowedOutcome(Schema):
    """The user selected one of the provided options."""

    option_id: str
    """The ID of the option the user selected."""

    outcome: Literal["selected"] = Field(default="selected", init=False)


class PermissionOption(AnnotatedObject):
    """An option presented to the user when requesting permission."""

    kind: PermissionKind
    """Hint about the nature of this permission option."""

    name: str
    """Human-readable label to display to the user."""

    option_id: str
    """Unique identifier for this permission option."""


ToolCallContent = (
    ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent
)
