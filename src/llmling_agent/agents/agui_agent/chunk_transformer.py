"""Chunk transformer for AG-UI events.

Transforms TEXT_MESSAGE_CHUNK and TOOL_CALL_CHUNK events into proper
START/CONTENT/END sequences for easier downstream processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ag_ui.core import (
    EventType,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from ag_ui.core import (
        BaseEvent,
        TextMessageChunkEvent,
        ToolCallChunkEvent,
    )


logger = get_logger(__name__)


class ChunkTransformer:
    """Transforms CHUNK events into proper START/CONTENT/END sequences.

    AG-UI supports two streaming modes:
    1. Explicit: TEXT_MESSAGE_START -> TEXT_MESSAGE_CONTENT* -> TEXT_MESSAGE_END
    2. Chunk: TEXT_MESSAGE_CHUNK* (implicit start/end based on message_id changes)

    This transformer normalizes chunk mode into explicit mode for consistent handling.
    Same applies for TOOL_CALL_CHUNK events.
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize transformer.

        Args:
            debug: Enable debug logging
        """
        self._debug = debug
        # Track active text message: message_id -> role
        self._active_text: dict[str, str] | None = None
        # Track active tool call: tool_call_id -> (name, parent_message_id)
        self._active_tool: dict[str, tuple[str, str | None]] | None = None

    def transform(self, event: BaseEvent) -> list[BaseEvent]:
        """Transform a single event, potentially expanding chunks.

        Args:
            event: Input event

        Returns:
            List of output events (may be empty, single, or multiple)
        """
        match event.type:
            case EventType.TEXT_MESSAGE_CHUNK:
                return self._handle_text_chunk(event)  # type: ignore[arg-type]

            case EventType.TOOL_CALL_CHUNK:
                return self._handle_tool_chunk(event)  # type: ignore[arg-type]

            # These events close any pending chunks
            case (
                EventType.TEXT_MESSAGE_START
                | EventType.TEXT_MESSAGE_END
                | EventType.TOOL_CALL_START
                | EventType.TOOL_CALL_END
                | EventType.RUN_FINISHED
                | EventType.RUN_ERROR
            ):
                close_events = self._close_all_pending()
                return [*close_events, event]

            case _:
                # Pass through other events unchanged
                return [event]

    def _handle_text_chunk(self, event: TextMessageChunkEvent) -> list[BaseEvent]:
        """Handle TEXT_MESSAGE_CHUNK event."""
        result: list[BaseEvent] = []
        message_id = event.message_id
        role = event.role or "assistant"
        delta = event.delta

        # Check if we need to close current text message (different ID)
        if self._active_text is not None:
            current_id = next(iter(self._active_text.keys()))
            if message_id and message_id != current_id:
                result.extend(self._close_text_message())

        # Start new message if needed
        if self._active_text is None and message_id:
            self._active_text = {message_id: role}
            start_event = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role=role,
            )
            result.append(start_event)
            if self._debug:
                logger.debug("Chunk transformer: TEXT_MESSAGE_START", message_id=message_id)

        # Emit content if we have delta and active message
        if delta and self._active_text:
            current_id = next(iter(self._active_text.keys()))
            content_event = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=current_id,
                delta=delta,
            )
            result.append(content_event)

        return result

    def _handle_tool_chunk(self, event: ToolCallChunkEvent) -> list[BaseEvent]:
        """Handle TOOL_CALL_CHUNK event."""
        result: list[BaseEvent] = []
        tool_call_id = event.tool_call_id
        tool_name = event.tool_call_name
        parent_id = event.parent_message_id
        delta = event.delta

        # Check if we need to close current tool call (different ID)
        if self._active_tool is not None:
            current_id = next(iter(self._active_tool.keys()))
            if tool_call_id and tool_call_id != current_id:
                result.extend(self._close_tool_call())

        # Start new tool call if needed
        if self._active_tool is None and tool_call_id and tool_name:
            self._active_tool = {tool_call_id: (tool_name, parent_id)}
            start_event = ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
                tool_call_name=tool_name,
                parent_message_id=parent_id,
            )
            result.append(start_event)
            if self._debug:
                logger.debug(
                    "Chunk transformer: TOOL_CALL_START",
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )

        # Emit args if we have delta and active tool call
        if delta and self._active_tool:
            current_id = next(iter(self._active_tool.keys()))
            args_event = ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=current_id,
                delta=delta,
            )
            result.append(args_event)

        return result

    def _close_text_message(self) -> list[BaseEvent]:
        """Close active text message."""
        if self._active_text is None:
            return []

        message_id = next(iter(self._active_text.keys()))
        self._active_text = None

        end_event = TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id,
        )
        if self._debug:
            logger.debug("Chunk transformer: TEXT_MESSAGE_END", message_id=message_id)
        return [end_event]

    def _close_tool_call(self) -> list[BaseEvent]:
        """Close active tool call."""
        if self._active_tool is None:
            return []

        tool_call_id = next(iter(self._active_tool.keys()))
        self._active_tool = None

        end_event = ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=tool_call_id,
        )
        if self._debug:
            logger.debug("Chunk transformer: TOOL_CALL_END", tool_call_id=tool_call_id)
        return [end_event]

    def _close_all_pending(self) -> list[BaseEvent]:
        """Close all pending chunks (text and tool)."""
        result: list[BaseEvent] = []
        result.extend(self._close_text_message())
        result.extend(self._close_tool_call())
        return result

    def flush(self) -> list[BaseEvent]:
        """Flush any pending events at end of stream.

        Call this when the stream ends to ensure all pending
        messages/tool calls are properly closed.

        Returns:
            List of closing events
        """
        return self._close_all_pending()

    def reset(self) -> None:
        """Reset transformer state."""
        self._active_text = None
        self._active_tool = None
