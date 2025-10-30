"""Event stream events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai import AgentStreamEvent

from llmling_agent.messaging.messages import ChatMessage  # noqa: TC001


@dataclass(kw_only=True)
class StreamCompleteEvent[TContent]:
    """Event indicating streaming is complete with final message."""

    message: ChatMessage[TContent]
    """The final chat message with all metadata."""
    event_kind: Literal["stream_complete"] = "stream_complete"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallProgressEvent:
    """Event indicating the tool call progress."""

    progress: int
    """The current progress of the tool call."""
    total: int
    """The total progress of the tool call."""
    message: str
    """Progress message."""
    tool_name: str
    """The name of the tool being called."""
    tool_call_id: str
    """The ID of the tool call."""
    tool_input: dict[str, Any] | None
    """The input provided to the tool."""


@dataclass(kw_only=True)
class CommandOutputEvent:
    """Event for slash command output."""

    command: str
    """The command name that was executed."""
    output: str
    """The output text from the command."""
    event_kind: Literal["command_output"] = "command_output"
    """Event type identifier."""


@dataclass(kw_only=True)
class CommandCompleteEvent:
    """Event indicating slash command execution is complete."""

    command: str
    """The command name that was completed."""
    success: bool
    """Whether the command executed successfully."""
    event_kind: Literal["command_complete"] = "command_complete"
    """Event type identifier."""


type RichAgentStreamEvent[OutputDataT] = (
    AgentStreamEvent | StreamCompleteEvent[OutputDataT] | ToolCallProgressEvent
)


type SlashedAgentStreamEvent[OutputDataT] = (
    RichAgentStreamEvent[OutputDataT] | CommandOutputEvent | CommandCompleteEvent
)
