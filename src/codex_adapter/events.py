"""Codex event types for streaming.

Uses discriminated unions with TypeAdapter for type-safe event parsing.
Each event type is a proper BaseModel with the event_type as the discriminator.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.alias_generators import to_camel

from codex_adapter.models import (
    AccountLoginCompletedData,
    AccountRateLimitsUpdatedData,
    AccountUpdatedData,
    AgentMessageDeltaData,
    AuthStatusChangeData,
    CommandExecutionOutputDeltaData,
    CommandExecutionTerminalInteractionData,
    DeprecationNoticeData,
    ErrorEventData,
    FileChangeOutputDeltaData,
    ItemCompletedData,
    ItemStartedData,
    LoginChatGptCompleteData,
    McpServerOAuthLoginCompletedData,
    McpToolCallProgressData,
    RawResponseItemCompletedData,
    ReasoningSummaryPartAddedData,
    ReasoningSummaryTextDeltaData,
    ReasoningTextDeltaData,
    SessionConfiguredData,
    ThreadCompactedData,
    ThreadStartedData,
    ThreadTokenUsageUpdatedData,
    TurnCompletedData,
    TurnDiffUpdatedData,
    TurnErrorData,
    TurnPlanUpdatedData,
    TurnStartedData,
    WindowsWorldWritableWarningData,
)


# ============================================================================
# Base event class
# ============================================================================


class CodexEventBase(BaseModel):
    """Base class for all Codex events.

    Each event has an event_type discriminator that maps to the JSON-RPC method.
    The data field contains the typed payload specific to that event type.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


# ============================================================================
# Error events
# ============================================================================


class ErrorEvent(CodexEventBase):
    """Error event from the Codex server."""

    event_type: Literal["error"] = "error"
    data: ErrorEventData


# ============================================================================
# Thread lifecycle events
# ============================================================================


class ThreadStartedEvent(CodexEventBase):
    """Thread started event."""

    event_type: Literal["thread/started"] = "thread/started"
    data: ThreadStartedData


class ThreadTokenUsageUpdatedEvent(CodexEventBase):
    """Thread token usage updated event."""

    event_type: Literal["thread/tokenUsage/updated"] = "thread/tokenUsage/updated"
    data: ThreadTokenUsageUpdatedData


class ThreadCompactedEvent(CodexEventBase):
    """Thread compacted event."""

    event_type: Literal["thread/compacted"] = "thread/compacted"
    data: ThreadCompactedData


# ============================================================================
# Turn lifecycle events
# ============================================================================


class TurnStartedEvent(CodexEventBase):
    """Turn started event."""

    event_type: Literal["turn/started"] = "turn/started"
    data: TurnStartedData


class TurnCompletedEvent(CodexEventBase):
    """Turn completed event."""

    event_type: Literal["turn/completed"] = "turn/completed"
    data: TurnCompletedData


class TurnErrorEvent(CodexEventBase):
    """Turn error event."""

    event_type: Literal["turn/error"] = "turn/error"
    data: TurnErrorData


class TurnDiffUpdatedEvent(CodexEventBase):
    """Turn diff updated event."""

    event_type: Literal["turn/diff/updated"] = "turn/diff/updated"
    data: TurnDiffUpdatedData


class TurnPlanUpdatedEvent(CodexEventBase):
    """Turn plan updated event."""

    event_type: Literal["turn/plan/updated"] = "turn/plan/updated"
    data: TurnPlanUpdatedData


# ============================================================================
# Item lifecycle events
# ============================================================================


class ItemStartedEvent(CodexEventBase):
    """Item started event."""

    event_type: Literal["item/started"] = "item/started"
    data: ItemStartedData


class ItemCompletedEvent(CodexEventBase):
    """Item completed event."""

    event_type: Literal["item/completed"] = "item/completed"
    data: ItemCompletedData


class RawResponseItemCompletedEvent(CodexEventBase):
    """Raw response item completed event."""

    event_type: Literal["rawResponseItem/completed"] = "rawResponseItem/completed"
    data: RawResponseItemCompletedData


# ============================================================================
# Item delta events - Agent messages
# ============================================================================


class AgentMessageDeltaEvent(CodexEventBase):
    """Agent message delta event (streaming text)."""

    event_type: Literal["item/agentMessage/delta"] = "item/agentMessage/delta"
    data: AgentMessageDeltaData


# ============================================================================
# Item delta events - Reasoning
# ============================================================================


class ReasoningSummaryTextDeltaEvent(CodexEventBase):
    """Reasoning summary text delta event."""

    event_type: Literal["item/reasoning/summaryTextDelta"] = "item/reasoning/summaryTextDelta"
    data: ReasoningSummaryTextDeltaData


class ReasoningSummaryPartAddedEvent(CodexEventBase):
    """Reasoning summary part added event."""

    event_type: Literal["item/reasoning/summaryPartAdded"] = "item/reasoning/summaryPartAdded"
    data: ReasoningSummaryPartAddedData


class ReasoningTextDeltaEvent(CodexEventBase):
    """Reasoning text delta event."""

    event_type: Literal["item/reasoning/textDelta"] = "item/reasoning/textDelta"
    data: ReasoningTextDeltaData


# ============================================================================
# Item delta events - Command execution
# ============================================================================


class CommandExecutionOutputDeltaEvent(CodexEventBase):
    """Command execution output delta event."""

    event_type: Literal["item/commandExecution/outputDelta"] = "item/commandExecution/outputDelta"
    data: CommandExecutionOutputDeltaData


class CommandExecutionTerminalInteractionEvent(CodexEventBase):
    """Command execution terminal interaction event."""

    event_type: Literal["item/commandExecution/terminalInteraction"] = (
        "item/commandExecution/terminalInteraction"
    )
    data: CommandExecutionTerminalInteractionData


# ============================================================================
# Item delta events - File changes
# ============================================================================


class FileChangeOutputDeltaEvent(CodexEventBase):
    """File change output delta event."""

    event_type: Literal["item/fileChange/outputDelta"] = "item/fileChange/outputDelta"
    data: FileChangeOutputDeltaData


# ============================================================================
# Item delta events - MCP tool calls
# ============================================================================


class McpToolCallProgressEvent(CodexEventBase):
    """MCP tool call progress event."""

    event_type: Literal["item/mcpToolCall/progress"] = "item/mcpToolCall/progress"
    data: McpToolCallProgressData


# ============================================================================
# MCP OAuth events
# ============================================================================


class McpServerOAuthLoginCompletedEvent(CodexEventBase):
    """MCP server OAuth login completed event."""

    event_type: Literal["mcpServer/oauthLogin/completed"] = "mcpServer/oauthLogin/completed"
    data: McpServerOAuthLoginCompletedData


# ============================================================================
# Account/Auth events
# ============================================================================


class AccountUpdatedEvent(CodexEventBase):
    """Account updated event."""

    event_type: Literal["account/updated"] = "account/updated"
    data: AccountUpdatedData


class AccountRateLimitsUpdatedEvent(CodexEventBase):
    """Account rate limits updated event."""

    event_type: Literal["account/rateLimits/updated"] = "account/rateLimits/updated"
    data: AccountRateLimitsUpdatedData


class AccountLoginCompletedEvent(CodexEventBase):
    """Account login completed event."""

    event_type: Literal["account/login/completed"] = "account/login/completed"
    data: AccountLoginCompletedData


class AuthStatusChangeEvent(CodexEventBase):
    """Auth status change event (legacy v1)."""

    event_type: Literal["authStatusChange"] = "authStatusChange"
    data: AuthStatusChangeData


class LoginChatGptCompleteEvent(CodexEventBase):
    """Login ChatGPT complete event (legacy v1)."""

    event_type: Literal["loginChatGptComplete"] = "loginChatGptComplete"
    data: LoginChatGptCompleteData


# ============================================================================
# System events
# ============================================================================


class SessionConfiguredEvent(CodexEventBase):
    """Session configured event."""

    event_type: Literal["sessionConfigured"] = "sessionConfigured"
    data: SessionConfiguredData


class DeprecationNoticeEvent(CodexEventBase):
    """Deprecation notice event."""

    event_type: Literal["deprecationNotice"] = "deprecationNotice"
    data: DeprecationNoticeData


class WindowsWorldWritableWarningEvent(CodexEventBase):
    """Windows world writable warning event."""

    event_type: Literal["windows/worldWritableWarning"] = "windows/worldWritableWarning"
    data: WindowsWorldWritableWarningData


# ============================================================================
# Discriminated union of all event types
# ============================================================================


CodexEvent = Annotated[
    # Error events
    ErrorEvent
    # Thread lifecycle
    | ThreadStartedEvent
    | ThreadTokenUsageUpdatedEvent
    | ThreadCompactedEvent
    # Turn lifecycle
    | TurnStartedEvent
    | TurnCompletedEvent
    | TurnErrorEvent
    | TurnDiffUpdatedEvent
    | TurnPlanUpdatedEvent
    # Item lifecycle
    | ItemStartedEvent
    | ItemCompletedEvent
    | RawResponseItemCompletedEvent
    # Item deltas - agent messages
    | AgentMessageDeltaEvent
    # Item deltas - reasoning
    | ReasoningSummaryTextDeltaEvent
    | ReasoningSummaryPartAddedEvent
    | ReasoningTextDeltaEvent
    # Item deltas - command execution
    | CommandExecutionOutputDeltaEvent
    | CommandExecutionTerminalInteractionEvent
    # Item deltas - file changes
    | FileChangeOutputDeltaEvent
    # Item deltas - MCP tool calls
    | McpToolCallProgressEvent
    # MCP OAuth
    | McpServerOAuthLoginCompletedEvent
    # Account/Auth events
    | AccountUpdatedEvent
    | AccountRateLimitsUpdatedEvent
    | AccountLoginCompletedEvent
    | AuthStatusChangeEvent
    | LoginChatGptCompleteEvent
    # System events
    | SessionConfiguredEvent
    | DeprecationNoticeEvent
    | WindowsWorldWritableWarningEvent,
    Field(discriminator="event_type"),
]


# TypeAdapter for parsing events
_codex_event_adapter: TypeAdapter[CodexEvent] = TypeAdapter(CodexEvent)


# ============================================================================
# Event type literals (for external use)
# ============================================================================


EventType = Literal[
    # Error events
    "error",
    # Thread lifecycle
    "thread/started",
    "thread/tokenUsage/updated",
    "thread/compacted",
    # Turn lifecycle
    "turn/started",
    "turn/completed",
    "turn/error",
    "turn/diff/updated",
    "turn/plan/updated",
    # Item lifecycle
    "item/started",
    "item/completed",
    "rawResponseItem/completed",
    # Item deltas - agent messages
    "item/agentMessage/delta",
    # Item deltas - reasoning
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/reasoning/textDelta",
    # Item deltas - command execution
    "item/commandExecution/outputDelta",
    "item/commandExecution/terminalInteraction",
    # Item deltas - file changes
    "item/fileChange/outputDelta",
    # Item deltas - MCP tool calls
    "item/mcpToolCall/progress",
    # MCP OAuth
    "mcpServer/oauthLogin/completed",
    # Account/Auth events
    "account/updated",
    "account/rateLimits/updated",
    "account/login/completed",
    "authStatusChange",
    "loginChatGptComplete",
    # System events
    "sessionConfigured",
    "deprecationNotice",
    "windows/worldWritableWarning",
]


# ============================================================================
# Factory function for creating events from JSON-RPC notifications
# ============================================================================


def parse_codex_event(method: str, params: dict[str, Any] | None) -> CodexEvent | None:
    """Create a CodexEvent from a JSON-RPC notification.

    Uses the TypeAdapter with discriminator for type-safe parsing of known events.
    Returns None for legacy codex/event/* methods (duplicates of V2 events).

    Args:
        method: The JSON-RPC notification method (event type)
        params: The notification parameters (event data)

    Returns:
        A typed CodexEvent instance, or None for legacy events to skip

    Raises:
        ValueError: If the event type is unknown (add a new model for it)
    """
    # Skip legacy V1 events - they duplicate V2 events in a different format
    if method.startswith("codex/event/"):
        return None

    event_data = {"event_type": method, "data": params or {}}
    return _codex_event_adapter.validate_python(event_data)


# ============================================================================
# Type-safe delta extraction
# ============================================================================


# Type alias for all delta events
DeltaEvent = (
    AgentMessageDeltaEvent
    | ReasoningTextDeltaEvent
    | ReasoningSummaryTextDeltaEvent
    | CommandExecutionOutputDeltaEvent
    | FileChangeOutputDeltaEvent
)


def get_text_delta(event: CodexEvent) -> str:
    """Extract text delta from a delta event.

    Type-safe extraction that only works on events with delta content.

    Args:
        event: Any CodexEvent

    Returns:
        The delta text if this is a delta event, empty string otherwise
    """
    match event:
        case (
            AgentMessageDeltaEvent(data=data)
            | ReasoningTextDeltaEvent(data=data)
            | ReasoningSummaryTextDeltaEvent(data=data)
            | CommandExecutionOutputDeltaEvent(data=data)
            | FileChangeOutputDeltaEvent(data=data)
        ):
            return data.delta
        case _:
            return ""


def is_delta_event(event: CodexEvent) -> bool:
    """Check if this is a delta event (streaming content)."""
    return isinstance(
        event,
        AgentMessageDeltaEvent
        | ReasoningTextDeltaEvent
        | ReasoningSummaryTextDeltaEvent
        | CommandExecutionOutputDeltaEvent
        | FileChangeOutputDeltaEvent,
    )


def is_completed_event(event: CodexEvent) -> bool:
    """Check if this is a completion event."""
    return isinstance(
        event,
        TurnCompletedEvent
        | ItemCompletedEvent
        | RawResponseItemCompletedEvent
        | McpServerOAuthLoginCompletedEvent
        | AccountLoginCompletedEvent
        | LoginChatGptCompleteEvent,
    )


def is_error_event(event: CodexEvent) -> bool:
    """Check if this is an error event."""
    return isinstance(event, ErrorEvent | TurnErrorEvent)
