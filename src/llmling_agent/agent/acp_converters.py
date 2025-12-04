"""ACP to native event converters.

This module provides conversion from ACP session updates to native llmling-agent
streaming events, enabling ACPAgent to yield the same event types as native agents.

This is the reverse of the conversion done in acp_server/session.py handle_event().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import PartDeltaEvent
from pydantic_ai.messages import TextPartDelta, ThinkingPartDelta

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from llmling_agent.agent.events import (
    DiffContentItem,
    FileEditProgressEvent,
    LocationContentItem,
    PlanUpdateEvent,
    ProcessStartEvent,
    TerminalContentItem,
    ToolCallContentItem,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from acp.schema import SessionUpdate
    from acp.schema.tool_call import (
        ToolCallContent,
        ToolCallLocation,
    )
    from llmling_agent.agent.events import RichAgentStreamEvent


def convert_acp_locations(
    locations: list[ToolCallLocation] | None,
) -> list[LocationContentItem]:
    """Convert ACP ToolCallLocation list to native LocationContentItem list."""
    if not locations:
        return []
    return [LocationContentItem(path=loc.path, line=loc.line) for loc in locations]


def convert_acp_content(
    content: list[ToolCallContent] | None,
) -> list[ToolCallContentItem]:
    """Convert ACP ToolCallContent list to native ToolCallContentItem list."""
    if not content:
        return []

    result: list[ToolCallContentItem] = []
    for item in content:
        match item.type:
            case "terminal":
                # TerminalToolCallContent
                result.append(TerminalContentItem(terminal_id=item.terminal_id))  # type: ignore[union-attr]
            case "diff":
                # FileEditToolCallContent
                result.append(
                    DiffContentItem(
                        path=item.path,  # type: ignore[union-attr]
                        old_text=item.old_text,  # type: ignore[union-attr]
                        new_text=item.new_text,  # type: ignore[union-attr]
                    )
                )
            case "content":
                # ContentToolCallContent - extract text if present
                if hasattr(item, "content") and isinstance(item.content, TextContentBlock):  # type: ignore[union-attr]
                    from llmling_agent.agent.events import TextContentItem

                    result.append(TextContentItem(text=item.content.text))  # type: ignore[union-attr]
    return result


def acp_to_native_event(update: SessionUpdate) -> RichAgentStreamEvent[Any] | None:
    """Convert ACP session update to native streaming event.

    Args:
        update: ACP SessionUpdate from session/update notification

    Returns:
        Corresponding native event, or None if no mapping exists
    """
    match update:
        # Text message chunks -> PartDeltaEvent with TextPartDelta
        case AgentMessageChunk(content=TextContentBlock(text=text)):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=text))

        # Thought chunks -> PartDeltaEvent with ThinkingPartDelta
        case AgentThoughtChunk(content=TextContentBlock(text=text)):
            return PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=text))

        # User message echo - could emit as PartStartEvent if needed
        case UserMessageChunk():
            return None  # Usually ignored

        # Tool call start -> ToolCallStartEvent
        case ToolCallStart() as tc:
            return ToolCallStartEvent(
                tool_call_id=tc.tool_call_id,
                tool_name=tc.title,  # ACP uses title, not separate tool_name
                title=tc.title,
                kind=tc.kind or "other",
                content=convert_acp_content(list(tc.content) if tc.content else None),
                locations=convert_acp_locations(list(tc.locations) if tc.locations else None),
                raw_input=tc.raw_input or {},
            )

        # Tool call progress -> ToolCallProgressEvent
        case ToolCallProgress() as tc:
            # Check for special content types
            items = convert_acp_content(list(tc.content) if tc.content else None)

            # Check if this is a file edit progress (has diff content)
            for item in items:
                if isinstance(item, DiffContentItem):
                    return FileEditProgressEvent(
                        path=item.path,
                        old_text=item.old_text or "",
                        new_text=item.new_text,
                        status=tc.status or "in_progress",  # type: ignore[arg-type]
                        tool_call_id=tc.tool_call_id,
                    )

            # Check if this is a process/terminal event
            for item in items:
                if isinstance(item, TerminalContentItem):
                    # This is a process-related update
                    return ProcessStartEvent(
                        process_id=item.terminal_id,
                        command=tc.title or "",
                        success=tc.status != "failed",
                        error=str(tc.raw_output) if tc.status == "failed" else None,
                        tool_call_id=tc.tool_call_id,
                    )

            # Generic tool call progress
            return ToolCallProgressEvent(
                tool_call_id=tc.tool_call_id,
                status=tc.status or "in_progress",
                title=tc.title,
                items=items,
                message=str(tc.raw_output) if tc.raw_output else None,
            )

        # Plan update -> PlanUpdateEvent
        case AgentPlanUpdate(entries=entries):
            from llmling_agent.resource_providers.plan_provider import (
                PlanEntry as NativePlanEntry,
            )

            native_entries = [
                NativePlanEntry(
                    content=e.content,
                    priority=e.priority,
                    status=e.status,
                )
                for e in entries
            ]
            return PlanUpdateEvent(entries=native_entries)

        case _:
            return None


def extract_text_from_update(update: SessionUpdate) -> str | None:
    """Extract plain text content from an ACP session update.

    Args:
        update: ACP SessionUpdate

    Returns:
        Text content if this is a text-bearing update, None otherwise
    """
    match update:
        case AgentMessageChunk(content=TextContentBlock(text=text)):
            return text
        case AgentThoughtChunk(content=TextContentBlock(text=text)):
            return text
        case _:
            return None


def is_text_update(update: SessionUpdate) -> bool:
    """Check if this update contains text content."""
    return extract_text_from_update(update) is not None
