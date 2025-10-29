"""Event stream events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
