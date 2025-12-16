"""AG-UI to native event converters.

This module provides conversion from AG-UI protocol events to native llmling-agent
streaming events, enabling AGUIAgent to yield the same event types as native agents.

Also provides conversion of llmling Tool objects to AG-UI Tool format for
client-side tool execution.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    ActivityDeltaEvent,
    ActivitySnapshotEvent,
    BinaryInputContent,
    CustomEvent as AGUICustomEvent,
    MessagesSnapshotEvent,
    RawEvent,
    RunErrorEvent as AGUIRunErrorEvent,
    RunStartedEvent as AGUIRunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextInputContent,
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
import anyenv
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    FileUrl,
    ImageUrl,
    PartDeltaEvent,
    VideoUrl,
)
from pydantic_ai.messages import CachePoint, TextPartDelta, ThinkingPartDelta

from llmling_agent.agents.events import (
    CustomEvent,
    PlanUpdateEvent,
    RunErrorEvent,
    RunStartedEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent as NativeToolCallStartEvent,
)
from llmling_agent.resource_providers.plan_provider import PlanEntry


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ag_ui.core import Event, InputContent, Tool as AGUITool
    from pydantic_ai import UserContent

    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.tools.base import Tool


def agui_to_native_event(event: Event) -> RichAgentStreamEvent[Any] | None:  # noqa: PLR0911
    """Convert AG-UI event to native streaming event.

    Args:
        event: AG-UI Event from SSE stream

    Returns:
        Corresponding native event, or None if no mapping exists
    """
    match event:
        # === Lifecycle Events ===

        case AGUIRunStartedEvent(thread_id=thread_id, run_id=run_id):
            return RunStartedEvent(thread_id=thread_id, run_id=run_id)

        case AGUIRunErrorEvent(message=message, code=code):
            return RunErrorEvent(message=message, code=code)

        # === Text Message Events ===

        case TextMessageContentEvent(delta=delta):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta))

        case TextMessageChunkEvent(delta=delta) if delta:
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta))

        case ThinkingTextMessageContentEvent(delta=delta):
            return PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=delta))

        case TextMessageStartEvent() | TextMessageEndEvent():
            return None

        case ThinkingTextMessageStartEvent() | ThinkingTextMessageEndEvent():
            return None
        # === Tool Call Events ===

        case ToolCallStartEvent(tool_call_id=str() as tc_id, tool_call_name=name):
            return NativeToolCallStartEvent(tool_call_id=tc_id, tool_name=name, title=name)

        case ToolCallChunkEvent(tool_call_id=str() as tc_id, tool_call_name=str() as name):
            return NativeToolCallStartEvent(tool_call_id=tc_id, tool_name=name, title=name)

        case ToolCallArgsEvent(tool_call_id=tool_call_id, delta=_):
            return ToolCallProgressEvent(tool_call_id=tool_call_id, status="in_progress")

        case ToolCallResultEvent(tool_call_id=tc_id, content=content, message_id=_):
            return ToolCallProgressEvent(tool_call_id=tc_id, status="completed", message=content)

        case ToolCallEndEvent(tool_call_id=tool_call_id):
            return ToolCallProgressEvent(tool_call_id=tool_call_id, status="completed")

        # === Activity Events -> PlanUpdateEvent ===

        case ActivitySnapshotEvent(
            message_id=_, activity_type=activity_type, content=content, replace=_
        ):
            # Map activity content to plan entries if it looks like a plan
            if activity_type.upper() == "PLAN" and isinstance(content, list):
                entries = _content_to_plan_entries(content)
                if entries:
                    return PlanUpdateEvent(entries=entries)
            # For other activity types, wrap as custom event
            return CustomEvent(
                event_data={"activity_type": activity_type, "content": content},
                event_type=f"activity_{activity_type.lower()}",
                source="ag-ui",
            )

        case ActivityDeltaEvent(message_id=_, activity_type=activity_type, patch=patch):
            return CustomEvent(
                event_data={"activity_type": activity_type, "patch": patch},
                event_type=f"activity_delta_{activity_type.lower()}",
                source="ag-ui",
            )

        # === State Management Events ===

        case StateSnapshotEvent(snapshot=snapshot):
            return CustomEvent(event_data=snapshot, event_type="state_snapshot", source="ag-ui")

        case StateDeltaEvent(delta=delta):
            return CustomEvent(event_data=delta, event_type="state_delta", source="ag-ui")

        case MessagesSnapshotEvent(messages=messages):
            data = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]
            return CustomEvent(event_data=data, event_type="messages_snapshot", source="ag-ui")

        # === Special Events ===

        case RawEvent(event=raw_event, source=source):
            return CustomEvent(event_data=raw_event, event_type="raw", source=source or "ag-ui")

        case AGUICustomEvent(name=name, value=value):
            return CustomEvent(event_data=value, event_type=name, source="ag-ui")

        case _:
            return None


def _content_to_plan_entries(content: list[Any]) -> list[PlanEntry]:
    """Convert AG-UI activity content to PlanEntry list.

    Args:
        content: List of plan items from ActivitySnapshotEvent

    Returns:
        List of PlanEntry objects
    """
    entries: list[PlanEntry] = []
    for item in content:
        if isinstance(item, dict):
            # Try to extract plan entry fields
            entry_content = item.get("content") or item.get("text") or item.get("description", "")
            priority = item.get("priority", "medium")
            status = item.get("status", "pending")

            # Normalize values
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"

            if entry_content:
                entry = PlanEntry(content=str(entry_content), priority=priority, status=status)
                entries.append(entry)
        elif isinstance(item, str):
            entries.append(PlanEntry(content=item, priority="medium", status="pending"))
    return entries


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


def to_agui_input_content(
    parts: UserContent | Sequence[UserContent] | None,
) -> list[InputContent]:
    """Convert pydantic-ai UserContent parts to AG-UI InputContent format.

    Args:
        parts: UserContent part(s) to convert (str, ImageUrl, BinaryContent, etc.)

    Returns:
        List of AG-UI InputContent items
    """
    if parts is None:
        return []

    # Normalize to list
    part_list = (
        [parts] if isinstance(parts, str | FileUrl | BinaryContent | CachePoint) else list(parts)
    )

    result: list[InputContent] = []
    for part in part_list:
        match part:
            case str() as text:
                result.append(TextInputContent(text=text))

            case ImageUrl(url=url, media_type=media_type):
                result.append(BinaryInputContent(url=str(url), mime_type=media_type))

            case AudioUrl(url=url, media_type=media_type):
                result.append(BinaryInputContent(url=str(url), mime_type=media_type))

            case DocumentUrl(url=url, media_type=media_type):
                result.append(BinaryInputContent(url=str(url), mime_type=media_type))

            case VideoUrl(url=url, media_type=media_type):
                result.append(BinaryInputContent(url=str(url), mime_type=media_type))

            case FileUrl(url=url, media_type=media_type):
                # Generic FileUrl fallback
                result.append(BinaryInputContent(url=str(url), mime_type=media_type))

            case BinaryImage(data=data, media_type=media_type):
                encoded = base64.b64encode(data).decode("utf-8")
                result.append(BinaryInputContent(data=encoded, mime_type=media_type))

            case BinaryContent(data=data, media_type=media_type):
                encoded = base64.b64encode(data).decode("utf-8")
                result.append(BinaryInputContent(data=encoded, mime_type=media_type))

            case CachePoint():
                # Cache points are markers, not actual content - skip
                pass

            case _:
                # Fallback: convert to string
                result.append(TextInputContent(text=str(part)))

    return result


def to_agui_tool(tool: Tool) -> AGUITool:
    """Convert llmling Tool to AG-UI Tool format.

    Args:
        tool: llmling Tool instance

    Returns:
        AG-UI Tool with JSON Schema parameters
    """
    from ag_ui.core import Tool as AGUITool

    schema = tool.schema
    func_schema = schema["function"]
    return AGUITool(
        name=func_schema["name"],
        description=func_schema.get("description", ""),
        parameters=func_schema.get("parameters", {"type": "object", "properties": {}}),
    )


class ToolCallAccumulator:
    """Accumulates streamed tool call arguments.

    AG-UI streams tool call arguments as deltas, this class accumulates them
    and provides the complete arguments when the tool call ends.
    """

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, Any]] = {}

    def start(self, tool_call_id: str, tool_name: str) -> None:
        """Start tracking a new tool call."""
        self._calls[tool_call_id] = {"name": tool_name, "args_buffer": ""}

    def add_args(self, tool_call_id: str, delta: str) -> None:
        """Add argument delta to a tool call."""
        if tool_call_id in self._calls:
            self._calls[tool_call_id]["args_buffer"] += delta

    def complete(self, tool_call_id: str) -> tuple[str, dict[str, Any]] | None:
        """Complete a tool call and return (tool_name, parsed_args).

        Returns:
            Tuple of (tool_name, args_dict) or None if call not found
        """
        if tool_call_id not in self._calls:
            return None

        call_data = self._calls.pop(tool_call_id)
        args_str = call_data["args_buffer"]
        try:
            args = anyenv.load_json(args_str) if args_str else {}
        except anyenv.JsonLoadError:
            args = {"raw": args_str}
        return call_data["name"], args

    def get_pending(self, tool_call_id: str) -> tuple[str, str] | None:
        """Get pending call data (tool_name, args_buffer) without completing."""
        if tool_call_id not in self._calls:
            return None
        data = self._calls[tool_call_id]
        return data["name"], data["args_buffer"]

    def clear(self) -> None:
        """Clear all pending tool calls."""
        self._calls.clear()
