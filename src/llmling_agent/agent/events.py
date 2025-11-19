"""Event stream events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import AgentStreamEvent

from llmling_agent.messaging import ChatMessage  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.resource_providers.plan_provider import PlanEntry


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


@dataclass(kw_only=True)
class ToolCallCompleteEvent:
    """Event indicating tool call is complete with both input and output."""

    tool_name: str
    """The name of the tool that was called."""
    tool_call_id: str
    """The ID of the tool call."""
    tool_input: dict[str, Any]
    """The input provided to the tool."""
    tool_result: Any
    """The result returned by the tool."""
    agent_name: str
    """The name of the agent that made the tool call."""
    message_id: str
    """The message ID associated with this tool call."""
    event_kind: Literal["tool_call_complete"] = "tool_call_complete"
    """Event type identifier."""


@dataclass(kw_only=True)
class CustomEvent[T]:
    """Generic custom event that can be emitted during tool execution."""

    event_data: T
    """The custom event data of any type."""
    event_type: str = "custom"
    """Type identifier for the custom event."""
    source: str | None = None
    """Optional source identifier (tool name, etc.)."""
    event_kind: Literal["custom"] = "custom"
    """Event type identifier."""


@dataclass(kw_only=True)
class PlanUpdateEvent:
    """Event indicating plan state has changed."""

    entries: list[PlanEntry]
    """Current plan entries."""
    tool_call_id: str | None = None
    """Tool call ID for ACP notifications."""
    event_kind: Literal["plan_update"] = "plan_update"
    """Event type identifier."""


@dataclass(kw_only=True)
class FileOperationEvent:
    """Event for filesystem operations."""

    operation: Literal["read", "write", "delete", "list", "edit"]
    """The filesystem operation performed."""
    path: str
    """The file/directory path that was operated on."""
    success: bool
    """Whether the operation completed successfully."""
    error: str | None = None
    """Error message if operation failed."""
    size: int | None = None
    """Size of file in bytes (for successful operations)."""
    tool_call_id: str | None = None
    """Tool call ID for ACP notifications."""

    # Rich ACP metadata
    title: str | None = None
    """Display title for the operation."""
    kind: str | None = None
    """Tool operation kind (edit, read, write, etc.)."""
    locations: list[str] = field(default_factory=list)
    """File paths affected by the operation."""
    raw_input: dict[str, Any] = field(default_factory=dict)
    """Original tool input arguments."""
    raw_output: Any = None
    """Tool result data for failed operations."""

    event_kind: Literal["file_operation"] = "file_operation"
    """Event type identifier."""


@dataclass(kw_only=True)
class FileEditProgressEvent:
    """Event for file edit progress with diff information."""

    path: str
    """The file path being edited."""
    old_text: str
    """Original file content."""
    new_text: str
    """New file content."""
    status: Literal["in_progress", "completed", "failed"]
    """Current status of the edit operation."""
    changed_lines: list[int] = field(default_factory=list)
    """Line numbers that were changed."""
    tool_call_id: str | None = None
    """Tool call ID for ACP notifications."""
    event_kind: Literal["file_edit_progress"] = "file_edit_progress"
    """Event type identifier."""


@dataclass(kw_only=True)
class ProcessStartEvent:
    """Event for process start operations."""

    process_id: str
    """Unique process identifier."""
    command: str
    """Command being executed."""
    args: list[str] = field(default_factory=list)
    """Command arguments."""
    cwd: str | None = None
    """Working directory."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables."""
    output_limit: int | None = None
    """Maximum bytes of output to retain."""
    success: bool = True
    """Whether the process started successfully."""
    error: str | None = None
    """Error message if start failed."""
    tool_call_id: str | None = None
    """Tool call ID for notifications."""

    event_kind: Literal["process_start"] = "process_start"
    """Event type identifier."""


@dataclass(kw_only=True)
class ProcessOutputEvent:
    """Event for process output updates."""

    process_id: str
    """Process identifier."""
    output: str
    """Process output (stdout/stderr combined)."""
    stdout: str | None = None
    """Standard output (if separated)."""
    stderr: str | None = None
    """Standard error (if separated)."""
    truncated: bool = False
    """Whether output was truncated due to limits."""
    tool_call_id: str | None = None
    """Tool call ID for notifications."""

    event_kind: Literal["process_output"] = "process_output"
    """Event type identifier."""


@dataclass(kw_only=True)
class ProcessExitEvent:
    """Event for process completion."""

    process_id: str
    """Process identifier."""
    exit_code: int
    """Process exit code."""
    success: bool
    """Whether the process completed successfully (exit_code == 0)."""
    final_output: str | None = None
    """Final process output."""
    truncated: bool = False
    """Whether output was truncated due to limits."""
    tool_call_id: str | None = None
    """Tool call ID for notifications."""

    event_kind: Literal["process_exit"] = "process_exit"
    """Event type identifier."""


@dataclass(kw_only=True)
class ProcessKillEvent:
    """Event for process termination."""

    process_id: str
    """Process identifier."""
    success: bool
    """Whether the process was successfully killed."""
    error: str | None = None
    """Error message if kill failed."""
    tool_call_id: str | None = None
    """Tool call ID for notifications."""

    event_kind: Literal["process_kill"] = "process_kill"
    """Event type identifier."""


@dataclass(kw_only=True)
class ProcessReleaseEvent:
    """Event for process resource cleanup."""

    process_id: str
    """Process identifier."""
    success: bool
    """Whether resources were successfully released."""
    error: str | None = None
    """Error message if release failed."""
    tool_call_id: str | None = None
    """Tool call ID for notifications."""

    event_kind: Literal["process_release"] = "process_release"
    """Event type identifier."""


type RichAgentStreamEvent[OutputDataT] = (
    AgentStreamEvent
    | StreamCompleteEvent[OutputDataT]
    | ToolCallProgressEvent
    | ToolCallCompleteEvent
    | PlanUpdateEvent
    | FileOperationEvent
    | FileEditProgressEvent
    | ProcessStartEvent
    | ProcessOutputEvent
    | ProcessExitEvent
    | ProcessKillEvent
    | ProcessReleaseEvent
    | CustomEvent[Any]
)


type SlashedAgentStreamEvent[OutputDataT] = (
    RichAgentStreamEvent[OutputDataT] | CommandOutputEvent | CommandCompleteEvent
)
