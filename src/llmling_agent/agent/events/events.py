"""Event stream events.

TODO: The specialized process and file events (ProcessStartEvent, ProcessExitEvent,
FileOperationEvent, etc.) are essentially domain-specific versions of
ToolCallProgressEvent. These could potentially be merged into a single, more flexible
ToolCallProgressEvent that carries rich content (terminals, diffs, locations) and
domain metadata. This would align better with the ACP protocol's tool call structure
and reduce event type proliferation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import AgentStreamEvent

from llmling_agent.messaging import ChatMessage  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.resource_providers.plan_provider import PlanEntry
    from llmling_agent.tools.base import ToolKind


# Lifecycle events (aligned with AG-UI protocol)


@dataclass(kw_only=True)
class RunStartedEvent:
    """Signals the start of an agent run."""

    thread_id: str
    """ID of the conversation thread (conversation_id)."""
    run_id: str
    """ID of the agent run (unique per request/response cycle)."""
    agent_name: str | None = None
    """Name of the agent starting the run."""
    event_kind: Literal["run_started"] = "run_started"
    """Event type identifier."""


@dataclass(kw_only=True)
class RunErrorEvent:
    """Signals an error during an agent run."""

    message: str
    """Error message."""
    code: str | None = None
    """Error code."""
    run_id: str | None = None
    """ID of the agent run that failed."""
    agent_name: str | None = None
    """Name of the agent that errored."""
    event_kind: Literal["run_error"] = "run_error"
    """Event type identifier."""


# Unified tool call content models (dataclass versions of ACP schema models)


@dataclass(kw_only=True)
class TerminalContentItem:
    """Embed a terminal for live output display."""

    type: Literal["terminal"] = "terminal"
    """Content type identifier."""
    terminal_id: str
    """The ID of the terminal being embedded."""


@dataclass(kw_only=True)
class DiffContentItem:
    """File modification shown as a diff."""

    type: Literal["diff"] = "diff"
    """Content type identifier."""
    path: str
    """The file path being modified."""
    old_text: str | None = None
    """The original content (None for new files)."""
    new_text: str
    """The new content after modification."""


@dataclass(kw_only=True)
class LocationContentItem:
    """A file location being accessed or modified."""

    type: Literal["location"] = "location"
    """Content type identifier."""
    path: str
    """The file path being accessed or modified."""
    line: int | None = None
    """Optional line number within the file."""


@dataclass(kw_only=True)
class TextContentItem:
    """Simple text content."""

    type: Literal["text"] = "text"
    """Content type identifier."""
    text: str
    """The text content."""


# Union type for all tool call content items
ToolCallContentItem = TerminalContentItem | DiffContentItem | LocationContentItem | TextContentItem


@dataclass(kw_only=True)
class StreamCompleteEvent[TContent]:
    """Event indicating streaming is complete with final message."""

    message: ChatMessage[TContent]
    """The final chat message with all metadata."""
    event_kind: Literal["stream_complete"] = "stream_complete"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallStartEvent:
    """Event indicating a tool call has started with rich ACP metadata."""

    tool_call_id: str
    """The ID of the tool call."""
    tool_name: str
    """The name of the tool being called."""
    title: str
    """Human-readable title describing what the tool is doing."""
    kind: ToolKind = "other"
    """Tool kind (read, edit, delete, move, search, execute, think, fetch, other)."""
    content: list[ToolCallContentItem] = field(default_factory=list)
    """Content produced by the tool call."""
    locations: list[LocationContentItem] = field(default_factory=list)
    """File locations affected by this tool call."""
    raw_input: dict[str, Any] = field(default_factory=dict)
    """The raw input parameters sent to the tool."""

    event_kind: Literal["tool_call_start"] = "tool_call_start"
    """Event type identifier."""


@dataclass(kw_only=True)
class ToolCallProgressEvent:
    """Enhanced tool call progress event with rich content support.

    This event can carry various types of rich content (terminals, diffs, locations, text)
    and maps directly to ACP tool call notifications. It serves as a unified replacement
    for specialized events like ProcessStartEvent, FileOperationEvent, etc.
    """

    tool_call_id: str
    """The ID of the tool call."""
    status: Literal["pending", "in_progress", "completed", "failed"] = "in_progress"
    """Current execution status."""
    title: str | None = None
    """Human-readable title describing the operation."""

    # Rich content items (unified content + locations)
    items: list[ToolCallContentItem] = field(default_factory=list)
    """Rich content items (terminals, diffs, locations, text)."""

    # Legacy fields for backwards compatibility
    progress: int | None = None
    """The current progress of the tool call."""
    total: int | None = None
    """The total progress of the tool call."""
    message: str | None = None
    """Progress message (falls back to TextContentItem)."""
    tool_name: str | None = None
    """The name of the tool being called."""
    tool_input: dict[str, Any] | None = None
    """The input provided to the tool."""

    event_kind: Literal["tool_call_progress"] = "tool_call_progress"
    """Event type identifier."""


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
    | RunStartedEvent
    | RunErrorEvent
    | ToolCallStartEvent
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
