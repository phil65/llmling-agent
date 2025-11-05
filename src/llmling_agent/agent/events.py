"""Event stream events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import AgentStreamEvent

from llmling_agent.messaging.messages import ChatMessage  # noqa: TC001


if TYPE_CHECKING:
    import asyncio


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


def create_queuing_progress_handler(queue: asyncio.Queue[ToolCallProgressEvent]):
    """Create progress handler that converts to ToolCallProgressEvent."""

    async def progress_handler(
        progress: float,
        total: float | None,
        message: str | None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        event = ToolCallProgressEvent(
            progress=int(progress) if progress is not None else 0,
            total=int(total) if total is not None else 100,
            message=message or "",
            tool_name=tool_name or "",
            tool_call_id=tool_call_id or "",
            tool_input=tool_input,
        )
        await queue.put(event)

    return progress_handler
