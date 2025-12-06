"""AG-UI to native event converters.

This module provides conversion from AG-UI protocol events to native llmling-agent
streaming events, enabling AGUIAgent to yield the same event types as native agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    ActivityDeltaEvent,
    ActivitySnapshotEvent,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from pydantic_ai import PartDeltaEvent
from pydantic_ai.messages import TextPartDelta, ThinkingPartDelta

from llmling_agent.agent.events import (
    ToolCallProgressEvent,
    ToolCallStartEvent as NativeToolCallStartEvent,
)


if TYPE_CHECKING:
    from ag_ui.core import (
        Event,
    )

    from llmling_agent.agent.events import RichAgentStreamEvent


def agui_to_native_event(event: Event) -> RichAgentStreamEvent[Any] | None:  # noqa: PLR0911
    """Convert AG-UI event to native streaming event.

    Args:
        event: AG-UI Event from SSE stream

    Returns:
        Corresponding native event, or None if no mapping exists
    """
    match event:
        # Text message content -> PartDeltaEvent with TextPartDelta
        case TextMessageContentEvent(delta=delta):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta))

        # Text message chunks -> PartDeltaEvent with TextPartDelta
        case TextMessageChunkEvent(delta=delta) if delta:
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta))

        # Thinking message content -> PartDeltaEvent with ThinkingPartDelta
        case ThinkingTextMessageContentEvent(delta=delta):
            return PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=delta))

        # Message start/end events - could be used for metadata tracking
        case TextMessageStartEvent() | TextMessageEndEvent():
            return None

        case ThinkingTextMessageStartEvent() | ThinkingTextMessageEndEvent():
            return None

        # Tool call start -> NativeToolCallStartEvent
        case ToolCallStartEvent(
            tool_call_id=tool_call_id, tool_call_name=tool_name, parent_message_id=_
        ):
            return NativeToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=tool_name,
                kind="other",
                content=[],
                locations=[],
                raw_input={},
            )

        # Tool call chunks -> NativeToolCallStartEvent if new, else progress
        case ToolCallChunkEvent(
            tool_call_id=tool_call_id,
            tool_call_name=tool_name,
            parent_message_id=_,
            delta=_,
        ) if tool_call_id and tool_name:
            # This is a new tool call
            return NativeToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=tool_name,
                kind="other",
                content=[],
                locations=[],
                raw_input={},
            )

        # Tool call args accumulation - treat as progress
        case ToolCallArgsEvent(tool_call_id=tool_call_id, delta=_):
            return ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                status="in_progress",
            )

        # Tool call result
        case ToolCallResultEvent(
            tool_call_id=tool_call_id,
            content=content,
            message_id=_,
        ):
            return ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                status="completed",
                message=content,
            )

        # Tool call end
        case ToolCallEndEvent(tool_call_id=tool_call_id):
            return ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                status="completed",
            )

        # Activity events - could map to plan or custom events
        case ActivitySnapshotEvent(message_id=_, activity_type=_, content=_, replace=_):
            # Could be used for plan updates if activity_type == "PLAN"
            return None

        case ActivityDeltaEvent(message_id=_, activity_type=_, patch=_):
            return None

        case _:
            return None


def extract_text_from_event(event: Event) -> str | None:
    """Extract plain text content from an AG-UI event.

    Args:
        event: AG-UI Event

    Returns:
        Text content if this is a text-bearing event, None otherwise
    """
    match event:
        case TextMessageContentEvent(delta=delta):
            return delta
        case TextMessageChunkEvent(delta=delta) if delta:
            return delta
        case ThinkingTextMessageContentEvent(delta=delta):
            return delta
        case _:
            return None


def is_text_event(event: Event) -> bool:
    """Check if this event contains text content."""
    return extract_text_from_event(event) is not None
