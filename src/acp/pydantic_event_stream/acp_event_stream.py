"""ACP Event Stream implementation for Pydantic AI agents.

This module provides the ACPEventStream class that inherits from Pydantic AI's
BaseEventStream to convert Pydantic AI agent events into ACP protocol events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import TYPE_CHECKING, Any

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    SessionNotification,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
)


if TYPE_CHECKING:
    from pydantic_ai.messages import (
        BuiltinToolCallPart,
        BuiltinToolReturnPart,
        FinalResultEvent,
        FunctionToolResultEvent,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ThinkingPartDelta,
        ToolCallPart,
        ToolCallPartDelta,
    )
    from pydantic_ai.run import AgentRunResultEvent
    from pydantic_ai.ui.event_stream import BaseEventStream


from acp.base import Schema


class ACPRequestData(Schema):
    """Simple request data for ACP event stream."""

    session_id: str
    """Session identifier for the ACP session."""


class ACPEventStream(BaseEventStream[ACPRequestData, SessionNotification, Any]):
    """ACP Event Stream for converting Pydantic AI events to ACP protocol events.

    This class inherits from Pydantic AI's BaseEventStream and implements the
    conversion logic to transform agent events into ACP SessionNotification events.
    """

    def __init__(self, request: ACPRequestData) -> None:
        """Initialize ACP event stream.

        Args:
            request: ACP request data containing session information
        """
        super().__init__(request)
        self._current_tool_call_id: str | None = None
        self._tool_call_counter = 0
        self._builtin_tool_call_ids: dict[str, str] = {}

    def encode_event(self, event: SessionNotification, accept: str | None = None) -> str:
        """Encode an ACP SessionNotification as JSON.

        Args:
            event: The ACP SessionNotification to encode
            accept: The accept header value (ignored for ACP)

        Returns:
            JSON-encoded string representation of the event
        """
        return event.model_dump_json(by_alias=True, exclude_none=True)

    async def before_stream(self) -> AsyncIterator[SessionNotification]:
        """Yield events before agent streaming starts."""
        # ACP doesn't require explicit start events
        return
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[SessionNotification]:
        """Yield events after agent streaming completes."""
        # ACP doesn't require explicit end events
        return
        yield  # Make this an async generator

    async def on_error(self, error: Exception) -> AsyncIterator[SessionNotification]:
        """Handle errors during streaming.

        Args:
            error: The error that occurred during streaming

        Yields:
            ACP error notification events
        """
        error_content = TextContentBlock(text=f"Error: {error}")
        error_chunk = AgentMessageChunk(content=error_content)
        yield SessionNotification(session_id=self.request.session_id, update=error_chunk)

    async def handle_text_start(
        self,
        part: TextPart,
        follows_text: bool = False,
    ) -> AsyncIterator[SessionNotification]:
        """Handle a TextPart at start.

        Args:
            part: The TextPart
            follows_text: Whether this part follows another text part

        Yields:
            ACP SessionNotification events
        """
        if part.content:
            content_block = TextContentBlock(text=part.content)
            chunk = AgentMessageChunk(content=content_block)
            yield SessionNotification(session_id=self.request.session_id, update=chunk)

    async def handle_text_delta(
        self, delta: TextPartDelta
    ) -> AsyncIterator[SessionNotification]:
        """Handle a TextPartDelta.

        Args:
            delta: The TextPartDelta containing content changes

        Yields:
            ACP SessionNotification events
        """
        if delta.content_delta:
            content_block = TextContentBlock(text=delta.content_delta)
            chunk = AgentMessageChunk(content=content_block)
            yield SessionNotification(session_id=self.request.session_id, update=chunk)

    async def handle_text_end(
        self, part: TextPart, followed_by_text: bool = False
    ) -> AsyncIterator[SessionNotification]:
        """Handle the end of a TextPart.

        Args:
            part: The TextPart that ended
            followed_by_text: Whether this is followed by another text part

        Yields:
            ACP SessionNotification events
        """
        # ACP doesn't require explicit text end events
        return
        yield  # Make this an async generator

    async def handle_thinking_start(
        self,
        part: ThinkingPart,
        follows_thinking: bool = False,
    ) -> AsyncIterator[SessionNotification]:
        """Handle a ThinkingPart at start.

        Args:
            part: The ThinkingPart
            follows_thinking: Whether this part follows another thinking part

        Yields:
            ACP SessionNotification events
        """
        if part.content:
            thought_chunk = AgentThoughtChunk(thought=part.content)
            yield SessionNotification(
                session_id=self.request.session_id, update=thought_chunk
            )

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[SessionNotification]:
        """Handle the end of a ThinkingPart.

        Args:
            part: The ThinkingPart that ended
            followed_by_thinking: Whether this is followed by another thinking part

        Yields:
            ACP SessionNotification events
        """
        # ACP doesn't require explicit thinking end events
        return
        yield  # Make this an async generator

    async def handle_thinking_delta(
        self, delta: ThinkingPartDelta
    ) -> AsyncIterator[SessionNotification]:
        """Handle a ThinkingPartDelta.

        Args:
            delta: The ThinkingPartDelta containing thought changes

        Yields:
            ACP SessionNotification events
        """
        if delta.content_delta:
            thought_chunk = AgentThoughtChunk(thought=delta.content_delta)
            yield SessionNotification(
                session_id=self.request.session_id, update=thought_chunk
            )

    async def handle_tool_call_start(
        self, part: ToolCallPart
    ) -> AsyncIterator[SessionNotification]:
        """Handle a ToolCallPart at start.

        Args:
            part: The tool call part

        Yields:
            ACP SessionNotification events
        """
        self._tool_call_counter += 1
        self._current_tool_call_id = f"tool_call_{self._tool_call_counter}"

        tool_start = ToolCallStart(
            tool_call_id=self._current_tool_call_id,
            tool_name=part.tool_name,
            input=part.args or {},
        )
        yield SessionNotification(session_id=self.request.session_id, update=tool_start)

    async def handle_tool_call_delta(
        self, delta: ToolCallPartDelta
    ) -> AsyncIterator[SessionNotification]:
        """Handle a ToolCallPartDelta.

        Args:
            delta: The ToolCallPartDelta containing argument changes

        Yields:
            ACP SessionNotification events
        """
        if self._current_tool_call_id and delta.args_delta:
            # Convert args_delta to string if it's not already
            args_str = (
                delta.args_delta
                if isinstance(delta.args_delta, str)
                else json.dumps(delta.args_delta)
            )

            progress = ToolCallProgress(
                tool_call_id=self._current_tool_call_id,
                progress=f"Building arguments: {args_str}",
            )
            yield SessionNotification(session_id=self.request.session_id, update=progress)

    async def handle_tool_call_end(
        self, part: ToolCallPart
    ) -> AsyncIterator[SessionNotification]:
        """Handle a ToolCallPart at end.

        Args:
            part: The tool call part

        Yields:
            ACP SessionNotification events
        """
        if self._current_tool_call_id:
            tool_update = ToolCallUpdate(
                tool_call_id=self._current_tool_call_id,
                status="completed",
            )
            yield SessionNotification(
                session_id=self.request.session_id, update=tool_update
            )
            # Clear the current tool call ID
            self._current_tool_call_id = None

    async def handle_builtin_tool_call_end(
        self, part: BuiltinToolCallPart
    ) -> AsyncIterator[SessionNotification]:
        """Handle a BuiltinToolCallPart at end.

        Args:
            part: The builtin tool call part

        Yields:
            ACP SessionNotification events
        """
        # Look up the actual tool call ID for this builtin tool
        builtin_tool_call_id = self._builtin_tool_call_ids.get(part.tool_call_id)
        if builtin_tool_call_id:
            tool_update = ToolCallUpdate(
                tool_call_id=builtin_tool_call_id,
                status="completed",
            )
            yield SessionNotification(
                session_id=self.request.session_id, update=tool_update
            )

    async def handle_builtin_tool_call_start(
        self, part: BuiltinToolCallPart
    ) -> AsyncIterator[SessionNotification]:
        """Handle a BuiltinToolCallPart at start.

        Args:
            part: The builtin tool call part

        Yields:
            ACP SessionNotification events
        """
        self._tool_call_counter += 1
        tool_call_id = f"builtin_tool_call_{self._tool_call_counter}"

        # Track the mapping from pydantic tool call ID to our ACP tool call ID
        self._builtin_tool_call_ids[part.tool_call_id] = tool_call_id

        tool_start = ToolCallStart(
            tool_call_id=tool_call_id, tool_name=part.tool_name, input=part.args or {}
        )
        yield SessionNotification(session_id=self.request.session_id, update=tool_start)

    async def handle_builtin_tool_return(
        self, part: BuiltinToolReturnPart
    ) -> AsyncIterator[SessionNotification]:
        """Handle a BuiltinToolReturnPart.

        Args:
            part: The BuiltinToolReturnPart

        Yields:
            ACP SessionNotification events
        """
        if part.content:
            content_block = TextContentBlock(text=str(part.content))
            chunk = AgentMessageChunk(content=content_block)
            yield SessionNotification(session_id=self.request.session_id, update=chunk)

    async def handle_function_tool_result(
        self, event: FunctionToolResultEvent
    ) -> AsyncIterator[SessionNotification]:
        """Handle a FunctionToolResultEvent.

        Args:
            event: The function tool result event

        Yields:
            ACP SessionNotification events
        """
        result = event.result
        if hasattr(result, "content") and result.content:
            content_str = str(result.content)
            content_block = TextContentBlock(text=content_str)
            chunk = AgentMessageChunk(content=content_block)
            yield SessionNotification(session_id=self.request.session_id, update=chunk)

    async def handle_final_result(
        self, event: FinalResultEvent
    ) -> AsyncIterator[SessionNotification]:
        """Handle a FinalResultEvent.

        Args:
            event: The final result event

        Yields:
            ACP SessionNotification events
        """
        # Final results are typically handled by the session manager
        # We could emit a completion notification here if needed
        return
        yield  # Make this an async generator

    async def handle_run_result(
        self, event: AgentRunResultEvent
    ) -> AsyncIterator[SessionNotification]:
        """Handle an AgentRunResultEvent.

        Args:
            event: The agent run result event

        Yields:
            ACP SessionNotification events
        """
        # Store the result for potential access by the session
        self.result = event.result

        # We could emit usage/completion information here if the ACP protocol supports it
        return
        yield  # Make this an async generator

    async def mark_tool_call_failed(
        self, tool_call_id: str, error_message: str
    ) -> AsyncIterator[SessionNotification]:
        """Mark a tool call as failed with an error message.

        Args:
            tool_call_id: The ID of the tool call that failed
            error_message: The error message to include

        Yields:
            ACP SessionNotification events
        """
        tool_update = ToolCallUpdate(
            tool_call_id=tool_call_id,
            status="failed",
        )
        yield SessionNotification(session_id=self.request.session_id, update=tool_update)

        # Also emit the error as a message
        error_content = TextContentBlock(text=f"Tool call failed: {error_message}")
        error_chunk = AgentMessageChunk(content=error_content)
        yield SessionNotification(session_id=self.request.session_id, update=error_chunk)


__all__ = [
    "ACPEventStream",
    "ACPRequestData",
]
