"""AG-UI to native event converters.

This module provides conversion from AG-UI protocol events to native agentpool
streaming events, enabling AGUIAgent to yield the same event types as native agents.

Also provides conversion of native Tool objects to AG-UI Tool format for
client-side tool execution.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    BinaryContent,
    FileUrl,
    TextPartDelta,
    ThinkingPartDelta,
)

from agentpool.agents.events import (
    CustomEvent,
    PartDeltaEvent,
    PlanUpdateEvent,
    RunErrorEvent,
    RunStartedEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent as NativeToolCallStartEvent,
)
from agentpool.resource_providers.plan_provider import PlanEntry


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ag_ui.core import BaseEvent, InputContent, Tool as AGUITool
    from pydantic_ai import UserContent

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.tools.base import Tool


def agui_to_native_event(event: BaseEvent) -> RichAgentStreamEvent[Any] | None:  # noqa: PLR0911
    """Convert AG-UI event to native streaming event.

    Args:
        event: AG-UI Event from SSE stream

    Returns:
        Corresponding native event, or None if no mapping exists
    """
    from ag_ui.core import (
        ActivityDeltaEvent,
        ActivitySnapshotEvent,
        CustomEvent as AGUICustomEvent,
        MessagesSnapshotEvent,
        RawEvent,
        RunErrorEvent as AGUIRunErrorEvent,
        RunStartedEvent as AGUIRunStartedEvent,
        StateDeltaEvent,
        StateSnapshotEvent,
        TextMessageChunkEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ThinkingTextMessageContentEvent,
        ThinkingTextMessageEndEvent,
        ThinkingTextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallChunkEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
    )

    match event:
        # === Lifecycle Events ===

        case AGUIRunStartedEvent(thread_id=thread_id, run_id=run_id):
            return RunStartedEvent(thread_id=thread_id, run_id=run_id)

        case AGUIRunErrorEvent(message=message, code=code):
            return RunErrorEvent(message=message, code=code)

        # === Text Message Events ===
        case TextMessageContentEvent(delta=delta) | TextMessageChunkEvent(delta=str() as delta):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta))

        case TextMessageStartEvent() | TextMessageEndEvent():
            return None

        # === Thinking/Reasoning Events ===

        case ThinkingTextMessageContentEvent(delta=delta):
            return PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=delta))

        case (
            ThinkingStartEvent()
            | ThinkingEndEvent()
            | ThinkingTextMessageStartEvent()
            | ThinkingTextMessageEndEvent()
        ):
            # These mark thinking blocks but don't carry content
            return None

        # === Tool Call Events ===

        case ToolCallStartEvent(tool_call_id=str() as tc_id, tool_call_name=name):
            return NativeToolCallStartEvent(tool_call_id=tc_id, tool_name=name, title=name)

        case ToolCallChunkEvent(tool_call_id=str() as tc_id, tool_call_name=str() as name):
            return NativeToolCallStartEvent(tool_call_id=tc_id, tool_name=name, title=name)

        case ToolCallArgsEvent(tool_call_id=tc_id, delta=_):
            return ToolCallProgressEvent(tool_call_id=tc_id, status="in_progress")

        case ToolCallResultEvent(tool_call_id=tc_id, content=content, message_id=_):
            return ToolCallProgressEvent(tool_call_id=tc_id, status="completed", message=content)

        case ToolCallEndEvent(tool_call_id=tc_id):
            return ToolCallProgressEvent(tool_call_id=tc_id, status="completed")

        # === Activity Events -> PlanUpdateEvent ===

        case ActivitySnapshotEvent(activity_type=activity_type, content=content):
            # Map activity content to plan entries if it looks like a plan
            if (
                activity_type.upper() == "PLAN"
                and isinstance(content, list)
                and (entries := _content_to_plan_entries(content))
            ):
                return PlanUpdateEvent(entries=entries)
            # For other activity types, wrap as custom event
            return CustomEvent(
                event_data={"activity_type": activity_type, "content": content},
                event_type=f"activity_{activity_type.lower()}",
                source="ag-ui",
            )

        case ActivityDeltaEvent(activity_type=activity_type, patch=patch):
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


def to_agui_input_content(parts: Sequence[UserContent]) -> list[InputContent]:
    """Convert pydantic-ai UserContent parts to AG-UI InputContent format."""
    from ag_ui.core import BinaryInputContent, TextInputContent

    result: list[InputContent] = []
    for part in parts:
        match part:
            case str() as text:
                result.append(TextInputContent(text=text))

            case FileUrl(url=url, media_type=media_type):
                result.append(BinaryInputContent(url=str(url), mime_type=media_type))

            case BinaryContent(data=data, media_type=media_type):
                encoded = base64.b64encode(data).decode("utf-8")
                result.append(BinaryInputContent(data=encoded, mime_type=media_type))
    return result


def to_agui_tool(tool: Tool) -> AGUITool:
    """Convert native Tool to AG-UI Tool format."""
    from ag_ui.core import Tool as AGUITool

    func_schema = tool.schema["function"]
    return AGUITool(
        name=func_schema["name"],
        description=func_schema.get("description", ""),
        parameters=func_schema.get("parameters", {"type": "object", "properties": {}}),
    )
