"""SSE event models."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field

from agentpool_server.opencode_server.models.app import Project  # noqa: TC001
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.message import (  # noqa: TC001
    AssistantMessage,
    UserMessage,
)
from agentpool_server.opencode_server.models.parts import Part  # noqa: TC001
from agentpool_server.opencode_server.models.session import Session, SessionStatus  # noqa: TC001


class EmptyProperties(OpenCodeBaseModel):
    """Empty properties object."""


class ServerConnectedEvent(OpenCodeBaseModel):
    """Server connected event."""

    type: Literal["server.connected"] = "server.connected"
    properties: EmptyProperties = Field(default_factory=EmptyProperties)


class SessionInfoProperties(OpenCodeBaseModel):
    """Session info wrapper for events."""

    info: Session


class SessionCreatedEvent(OpenCodeBaseModel):
    """Session created event."""

    type: Literal["session.created"] = "session.created"
    properties: SessionInfoProperties

    @classmethod
    def create(cls, session: Session) -> Self:
        return cls(properties=SessionInfoProperties(info=session))


class SessionUpdatedEvent(OpenCodeBaseModel):
    """Session updated event."""

    type: Literal["session.updated"] = "session.updated"
    properties: SessionInfoProperties

    @classmethod
    def create(cls, session: Session) -> Self:
        return cls(properties=SessionInfoProperties(info=session))


class SessionDeletedProperties(OpenCodeBaseModel):
    """Properties for session deleted event."""

    session_id: str


class SessionDeletedEvent(OpenCodeBaseModel):
    """Session deleted event."""

    type: Literal["session.deleted"] = "session.deleted"
    properties: SessionDeletedProperties

    @classmethod
    def create(cls, session_id: str) -> Self:
        return cls(properties=SessionDeletedProperties(session_id=session_id))


class SessionStatusProperties(OpenCodeBaseModel):
    """Properties for session status event."""

    session_id: str
    status: SessionStatus


class SessionStatusEvent(OpenCodeBaseModel):
    """Session status event."""

    type: Literal["session.status"] = "session.status"
    properties: SessionStatusProperties

    @classmethod
    def create(cls, session_id: str, status: SessionStatus) -> Self:
        return cls(properties=SessionStatusProperties(session_id=session_id, status=status))


class SessionIdleProperties(OpenCodeBaseModel):
    """Properties for session idle event (deprecated but still used by TUI)."""

    session_id: str


class SessionIdleEvent(OpenCodeBaseModel):
    """Session idle event (deprecated but still used by TUI run command)."""

    type: Literal["session.idle"] = "session.idle"
    properties: SessionIdleProperties

    @classmethod
    def create(cls, session_id: str) -> Self:
        return cls(properties=SessionIdleProperties(session_id=session_id))


class SessionCompactedProperties(OpenCodeBaseModel):
    """Properties for session compacted event."""

    session_id: str


class SessionCompactedEvent(OpenCodeBaseModel):
    """Session compacted event - emitted when context compaction completes."""

    type: Literal["session.compacted"] = "session.compacted"
    properties: SessionCompactedProperties

    @classmethod
    def create(cls, session_id: str) -> Self:
        return cls(properties=SessionCompactedProperties(session_id=session_id))


class SessionErrorInfo(OpenCodeBaseModel):
    """Error information for session error event.

    Simplified version of OpenCode's error types (ProviderAuthError, UnknownError, etc.)
    """

    name: str
    """Error type name (e.g., 'UnknownError', 'ProviderAuthError')."""

    data: dict[str, Any] | None = None
    """Additional error data, typically contains 'message' field."""


class SessionErrorProperties(OpenCodeBaseModel):
    """Properties for session error event."""

    session_id: str | None = Field(default=None)
    error: SessionErrorInfo | None = None


class SessionErrorEvent(OpenCodeBaseModel):
    """Session error event - emitted when an error occurs during message processing."""

    type: Literal["session.error"] = "session.error"
    properties: SessionErrorProperties

    @classmethod
    def from_exception(cls, exception: Exception, session_id: str | None = None) -> Self:
        error_name = type(exception).__name__
        error_message = str(exception)
        return cls.create(session_id=session_id, error_name=error_name, error_message=error_message)

    @classmethod
    def create(
        cls,
        session_id: str | None = None,
        error_name: str = "UnknownError",
        error_message: str | None = None,
    ) -> Self:
        error_data = {"message": error_message} if error_message else None
        error = SessionErrorInfo(name=error_name, data=error_data)
        props = SessionErrorProperties(session_id=session_id, error=error)
        return cls(properties=props)


class MessageUpdatedEventProperties(OpenCodeBaseModel):
    """Properties for message updated event."""

    info: UserMessage | AssistantMessage


class MessageUpdatedEvent(OpenCodeBaseModel):
    """Message updated event."""

    type: Literal["message.updated"] = "message.updated"
    properties: MessageUpdatedEventProperties

    @classmethod
    def create(cls, message: UserMessage | AssistantMessage) -> Self:
        return cls(properties=MessageUpdatedEventProperties(info=message))


class PartUpdatedEventProperties(OpenCodeBaseModel):
    """Properties for part updated event."""

    part: Part
    delta: str | None = None


class PartUpdatedEvent(OpenCodeBaseModel):
    """Part updated event."""

    type: Literal["message.part.updated"] = "message.part.updated"
    properties: PartUpdatedEventProperties

    @classmethod
    def create(cls, part: Part, delta: str | None = None) -> Self:
        return cls(properties=PartUpdatedEventProperties(part=part, delta=delta))


class MessageRemovedProperties(OpenCodeBaseModel):
    """Properties for message removed event."""

    session_id: str
    message_id: str


class MessageRemovedEvent(OpenCodeBaseModel):
    """Message removed event - emitted during revert."""

    type: Literal["message.removed"] = "message.removed"
    properties: MessageRemovedProperties

    @classmethod
    def create(cls, session_id: str, message_id: str) -> Self:
        """Create message removed event."""
        props = MessageRemovedProperties(session_id=session_id, message_id=message_id)
        return cls(properties=props)


class PartRemovedProperties(OpenCodeBaseModel):
    """Properties for part removed event."""

    session_id: str
    message_id: str
    part_id: str


class PartRemovedEvent(OpenCodeBaseModel):
    """Part removed event - emitted during revert."""

    type: Literal["message.part.removed"] = "message.part.removed"
    properties: PartRemovedProperties

    @classmethod
    def create(cls, session_id: str, message_id: str, part_id: str) -> Self:
        """Create part removed event."""
        props = PartRemovedProperties(session_id=session_id, message_id=message_id, part_id=part_id)
        return cls(properties=props)


class PermissionToolInfo(OpenCodeBaseModel):
    """Tool information for permission event."""

    message_id: str
    """Message ID."""

    call_id: str | None = None
    """Optional tool call ID."""


class PermissionAskedProperties(OpenCodeBaseModel):
    """Properties for permission.asked event.

    Matches OpenCode's PermissionNext.Event.Asked schema.
    """

    id: str
    """Permission request ID."""

    session_id: str
    """Session ID."""

    permission: str
    """Tool/permission type name."""

    patterns: list[str]
    """Patterns for matching (e.g., file paths, commands)."""

    metadata: dict[str, Any]
    """Arbitrary metadata about the tool call."""

    always: list[str]
    """Patterns that would be approved for future requests if user selects 'always'."""

    tool: PermissionToolInfo
    """Tool call information."""


class PermissionRequestEvent(OpenCodeBaseModel):
    """Permission request event - sent when a tool needs user confirmation.

    Uses 'permission.asked' event type for OpenCode TUI compatibility.
    """

    type: Literal["permission.asked"] = "permission.asked"
    properties: PermissionAskedProperties

    @classmethod
    def create(
        cls,
        session_id: str,
        permission_id: str,
        tool_name: str,
        args_preview: str,
        message: str,
        message_id: str = "",
        call_id: str | None = None,
    ) -> Self:
        # Create pattern from tool name and args
        pattern = f"{tool_name}: {args_preview}" if args_preview else tool_name

        props = PermissionAskedProperties(
            id=permission_id,
            session_id=session_id,
            permission=tool_name,
            patterns=[pattern],
            metadata={"args_preview": args_preview},
            always=[pattern],  # Same pattern for "always" approval
            tool=PermissionToolInfo(message_id=message_id, call_id=call_id),
        )
        return cls(properties=props)


class PermissionRepliedProperties(OpenCodeBaseModel):
    """Properties for permission replied event.

    Matches OpenCode's permission.replied event schema.
    """

    session_id: str
    """Session ID."""

    request_id: str
    """Request/Permission ID."""

    reply: Literal["once", "always", "reject"]
    """Reply: 'once' | 'always' | 'reject'."""


class PermissionResolvedEvent(OpenCodeBaseModel):
    """Permission resolved event - sent when a permission request is answered.

    Uses 'permission.replied' event type for OpenCode TUI compatibility.
    """

    type: Literal["permission.replied"] = "permission.replied"
    properties: PermissionRepliedProperties

    @classmethod
    def create(
        cls,
        session_id: str,
        request_id: str,
        reply: Literal["once", "always", "reject"],
    ) -> Self:
        props = PermissionRepliedProperties(
            session_id=session_id,
            request_id=request_id,
            reply=reply,
        )
        return cls(properties=props)


# =============================================================================
# TUI Events - for external control of the TUI (e.g., VSCode extension)
# =============================================================================


class TuiPromptAppendProperties(OpenCodeBaseModel):
    """Properties for TUI prompt append event."""

    text: str


class TuiPromptAppendEvent(OpenCodeBaseModel):
    """TUI prompt append event - appends text to the prompt input."""

    type: Literal["tui.prompt.append"] = "tui.prompt.append"
    properties: TuiPromptAppendProperties

    @classmethod
    def create(cls, text: str) -> Self:
        return cls(properties=TuiPromptAppendProperties(text=text))


class TuiCommandExecuteProperties(OpenCodeBaseModel):
    """Properties for TUI command execute event."""

    command: str


class TuiCommandExecuteEvent(OpenCodeBaseModel):
    """TUI command execute event - executes a TUI command.

    Commands include:
    - session.list, session.new, session.share, session.interrupt, session.compact
    - session.page.up, session.page.down, session.half.page.up, session.half.page.down
    - session.first, session.last
    - prompt.clear, prompt.submit
    - agent.cycle
    """

    type: Literal["tui.command.execute"] = "tui.command.execute"
    properties: TuiCommandExecuteProperties

    @classmethod
    def create(cls, command: str) -> Self:
        return cls(properties=TuiCommandExecuteProperties(command=command))


class TuiToastShowProperties(OpenCodeBaseModel):
    """Properties for TUI toast show event."""

    title: str | None = None
    message: str
    variant: Literal["info", "success", "warning", "error"] = "info"
    duration: int = 5000  # Duration in milliseconds


class TuiToastShowEvent(OpenCodeBaseModel):
    """TUI toast show event - shows a toast notification."""

    type: Literal["tui.toast.show"] = "tui.toast.show"
    properties: TuiToastShowProperties

    @classmethod
    def create(
        cls,
        message: str,
        variant: Literal["info", "success", "warning", "error"] = "info",
        title: str | None = None,
        duration: int = 5000,
    ) -> Self:
        props = TuiToastShowProperties(
            title=title,
            message=message,
            variant=variant,
            duration=duration,
        )
        return cls(properties=props)


# =============================================================================
# Todo Events
# =============================================================================


class Todo(OpenCodeBaseModel):
    """A single todo item."""

    id: str
    """Unique identifier for the todo item."""

    content: str
    """Brief description of the task."""

    status: str
    """Current status: pending, in_progress, completed, cancelled."""

    priority: str
    """Priority level: high, medium, low."""


class TodoUpdatedProperties(OpenCodeBaseModel):
    """Properties for todo updated event."""

    session_id: str
    todos: list[Todo]


class TodoUpdatedEvent(OpenCodeBaseModel):
    """Todo list updated event."""

    type: Literal["todo.updated"] = "todo.updated"
    properties: TodoUpdatedProperties

    @classmethod
    def create(cls, session_id: str, todos: list[Todo]) -> Self:
        return cls(properties=TodoUpdatedProperties(session_id=session_id, todos=todos))


# =============================================================================
# File Watcher Events
# =============================================================================


class FileWatcherUpdatedProperties(OpenCodeBaseModel):
    """Properties for file watcher updated event."""

    file: str
    """Absolute path to the file that changed."""

    event: Literal["add", "change", "unlink"]
    """Type of change: add (created), change (modified), unlink (deleted)."""


class FileWatcherUpdatedEvent(OpenCodeBaseModel):
    """File watcher updated event - sent when a project file changes."""

    type: Literal["file.watcher.updated"] = "file.watcher.updated"
    properties: FileWatcherUpdatedProperties

    @classmethod
    def create(cls, file: str, event: Literal["add", "change", "unlink"]) -> Self:
        return cls(properties=FileWatcherUpdatedProperties(file=file, event=event))


# =============================================================================
# PTY Events
# =============================================================================


class PtyCreatedProperties(OpenCodeBaseModel):
    """Properties for PTY created event."""

    info: Any
    """PTY session info."""


class PtyCreatedEvent(OpenCodeBaseModel):
    """PTY session created event."""

    type: Literal["pty.created"] = "pty.created"
    properties: PtyCreatedProperties

    @classmethod
    def create(cls, info: Any) -> Self:
        return cls(properties=PtyCreatedProperties(info=info))


class PtyUpdatedProperties(OpenCodeBaseModel):
    """Properties for PTY updated event."""

    info: Any
    """PTY session info."""


class PtyUpdatedEvent(OpenCodeBaseModel):
    """PTY session updated event."""

    type: Literal["pty.updated"] = "pty.updated"
    properties: PtyUpdatedProperties

    @classmethod
    def create(cls, info: Any) -> Self:
        return cls(properties=PtyUpdatedProperties(info=info))


class PtyExitedProperties(OpenCodeBaseModel):
    """Properties for PTY exited event."""

    id: str
    """PTY session ID."""

    exit_code: int
    """Process exit code."""


class PtyExitedEvent(OpenCodeBaseModel):
    """PTY process exited event."""

    type: Literal["pty.exited"] = "pty.exited"
    properties: PtyExitedProperties

    @classmethod
    def create(cls, pty_id: str, exit_code: int) -> Self:
        return cls(properties=PtyExitedProperties(id=pty_id, exit_code=exit_code))


class PtyDeletedProperties(OpenCodeBaseModel):
    """Properties for PTY deleted event."""

    id: str
    """PTY session ID."""


class PtyDeletedEvent(OpenCodeBaseModel):
    """PTY session deleted event."""

    type: Literal["pty.deleted"] = "pty.deleted"
    properties: PtyDeletedProperties

    @classmethod
    def create(cls, pty_id: str) -> Self:
        return cls(properties=PtyDeletedProperties(id=pty_id))


# =============================================================================
# LSP Events
# =============================================================================


class LspStatus(OpenCodeBaseModel):
    """LSP server status information."""

    id: str
    """Server identifier (e.g., 'pyright', 'rust-analyzer')."""

    name: str
    """Server name."""

    root: str
    """Relative workspace root path."""

    status: Literal["connected", "error"]
    """Connection status."""


class LspUpdatedEvent(OpenCodeBaseModel):
    """LSP status updated event - sent when LSP server status changes."""

    type: Literal["lsp.updated"] = "lsp.updated"
    properties: EmptyProperties = Field(default_factory=EmptyProperties)


class LspClientDiagnosticsProperties(OpenCodeBaseModel):
    """Properties for LSP client diagnostics event."""

    server_id: str
    """LSP server ID that produced the diagnostics."""

    path: str
    """File path the diagnostics apply to."""


class LspClientDiagnosticsEvent(OpenCodeBaseModel):
    """LSP client diagnostics event - sent when diagnostics are published."""

    type: Literal["lsp.client.diagnostics"] = "lsp.client.diagnostics"
    properties: LspClientDiagnosticsProperties

    @classmethod
    def create(cls, server_id: str, path: str) -> Self:
        return cls(properties=LspClientDiagnosticsProperties(server_id=server_id, path=path))


# =============================================================================
# VCS Events
# =============================================================================


class ProjectUpdatedEvent(OpenCodeBaseModel):
    """Project metadata updated event."""

    type: Literal["project.updated"] = "project.updated"
    properties: Project

    @classmethod
    def create(cls, project: Project) -> Self:
        """Create project updated event."""
        return cls(properties=project)


class VcsBranchUpdatedProperties(OpenCodeBaseModel):
    """Properties for VCS branch updated event."""

    branch: str | None = None
    """Current branch name, or None if detached HEAD."""


class VcsBranchUpdatedEvent(OpenCodeBaseModel):
    """VCS branch updated event - sent when git branch changes."""

    type: Literal["vcs.branch.updated"] = "vcs.branch.updated"
    properties: VcsBranchUpdatedProperties

    @classmethod
    def create(cls, branch: str | None) -> Self:
        return cls(properties=VcsBranchUpdatedProperties(branch=branch))


class QuestionAskedProperties(OpenCodeBaseModel):
    """Properties for question asked event."""

    id: str
    session_id: str
    questions: list[dict[str, Any]]
    tool: dict[str, str] | None = None


class QuestionAskedEvent(OpenCodeBaseModel):
    """Question asked event - sent when agent asks a question."""

    type: Literal["question.asked"] = "question.asked"
    properties: QuestionAskedProperties

    @classmethod
    def create(
        cls,
        request_id: str,
        session_id: str,
        questions: list[dict[str, Any]],
        tool: dict[str, str] | None = None,
    ) -> Self:
        props = QuestionAskedProperties(
            id=request_id,
            session_id=session_id,
            questions=questions,
            tool=tool,
        )
        return cls(properties=props)


class QuestionRepliedProperties(OpenCodeBaseModel):
    """Properties for question replied event."""

    session_id: str
    request_id: str
    answers: list[list[str]]


class QuestionRepliedEvent(OpenCodeBaseModel):
    """Question replied event - sent when user answers a question."""

    type: Literal["question.replied"] = "question.replied"
    properties: QuestionRepliedProperties

    @classmethod
    def create(
        cls,
        session_id: str,
        request_id: str,
        answers: list[list[str]],
    ) -> Self:
        props = QuestionRepliedProperties(
            session_id=session_id,
            request_id=request_id,
            answers=answers,
        )
        return cls(properties=props)


class QuestionRejectedProperties(OpenCodeBaseModel):
    """Properties for question rejected event."""

    session_id: str
    request_id: str


class QuestionRejectedEvent(OpenCodeBaseModel):
    """Question rejected event - sent when user dismisses a question."""

    type: Literal["question.rejected"] = "question.rejected"
    properties: QuestionRejectedProperties

    @classmethod
    def create(
        cls,
        session_id: str,
        request_id: str,
    ) -> Self:
        props = QuestionRejectedProperties(session_id=session_id, request_id=request_id)
        return cls(properties=props)


Event = (
    ServerConnectedEvent
    | SessionCreatedEvent
    | SessionUpdatedEvent
    | SessionDeletedEvent
    | SessionStatusEvent
    | SessionErrorEvent
    | SessionIdleEvent
    | MessageUpdatedEvent
    | MessageRemovedEvent
    | PartUpdatedEvent
    | PartRemovedEvent
    | PermissionRequestEvent
    | PermissionResolvedEvent
    | QuestionAskedEvent
    | QuestionRepliedEvent
    | QuestionRejectedEvent
    | TodoUpdatedEvent
    | FileWatcherUpdatedEvent
    | SessionCompactedEvent
    | PtyCreatedEvent
    | PtyUpdatedEvent
    | PtyExitedEvent
    | PtyDeletedEvent
    | LspUpdatedEvent
    | LspClientDiagnosticsEvent
    | ProjectUpdatedEvent
    | VcsBranchUpdatedEvent
    | TuiPromptAppendEvent
    | TuiCommandExecuteEvent
    | TuiToastShowEvent
)
