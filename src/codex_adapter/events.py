"""Codex event types for streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from codex_adapter.models import EventData


# Event types from app-server notifications (complete list from schema)
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


@dataclass
class CodexEvent:
    """A streaming event from the Codex app-server.

    Attributes:
        event_type: The notification method (e.g., "item/agentMessage/delta")
        data: Typed event payload specific to the event type
        raw: The full JSON-RPC notification message
    """

    event_type: EventType
    data: EventData
    raw: dict[str, Any]

    @classmethod
    def from_notification(cls, method: EventType, params: dict[str, Any] | None) -> CodexEvent:
        """Create event from JSON-RPC notification with proper type routing."""
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
            GenericEventData,
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

        raw_params = params or {}

        # Map event type to data model - exact mapping from app-server-protocol
        data_class: type[EventData] = {  # type: ignore[assignment]
            # Error events
            "error": ErrorEventData,
            # Thread lifecycle
            "thread/started": ThreadStartedData,
            "thread/tokenUsage/updated": ThreadTokenUsageUpdatedData,
            "thread/compacted": ThreadCompactedData,
            # Turn lifecycle
            "turn/started": TurnStartedData,
            "turn/completed": TurnCompletedData,
            "turn/error": TurnErrorData,
            "turn/diff/updated": TurnDiffUpdatedData,
            "turn/plan/updated": TurnPlanUpdatedData,
            # Item lifecycle
            "item/started": ItemStartedData,
            "item/completed": ItemCompletedData,
            "rawResponseItem/completed": RawResponseItemCompletedData,
            # Item deltas - agent messages
            "item/agentMessage/delta": AgentMessageDeltaData,
            # Item deltas - reasoning
            "item/reasoning/textDelta": ReasoningTextDeltaData,
            "item/reasoning/summaryTextDelta": ReasoningSummaryTextDeltaData,
            "item/reasoning/summaryPartAdded": ReasoningSummaryPartAddedData,
            # Item deltas - command execution
            "item/commandExecution/outputDelta": CommandExecutionOutputDeltaData,
            "item/commandExecution/terminalInteraction": CommandExecutionTerminalInteractionData,
            # Item deltas - file changes
            "item/fileChange/outputDelta": FileChangeOutputDeltaData,
            # Item deltas - MCP tool calls
            "item/mcpToolCall/progress": McpToolCallProgressData,
            # MCP OAuth
            "mcpServer/oauthLogin/completed": McpServerOAuthLoginCompletedData,
            # Account/Auth events
            "account/updated": AccountUpdatedData,
            "account/rateLimits/updated": AccountRateLimitsUpdatedData,
            "account/login/completed": AccountLoginCompletedData,
            "authStatusChange": AuthStatusChangeData,
            "loginChatGptComplete": LoginChatGptCompleteData,
            # System events
            "sessionConfigured": SessionConfiguredData,
            "deprecationNotice": DeprecationNoticeData,
            "windows/worldWritableWarning": WindowsWorldWritableWarningData,
        }.get(method, GenericEventData)

        return cls(
            event_type=method,
            data=data_class.model_validate(raw_params),
            raw={"method": method, "params": params},
        )

    def is_delta(self) -> bool:
        """Check if this is a delta event (streaming content)."""
        return "delta" in self.event_type.lower()

    def is_completed(self) -> bool:
        """Check if this is a completion event."""
        return "completed" in self.event_type.lower()

    def is_error(self) -> bool:
        """Check if this is an error event."""
        return "error" in self.event_type.lower()

    def get_text_delta(self) -> str:
        """Extract text delta from message/command events.

        Different event types use different field names:
        - agentMessage/delta: delta field
        - commandExecution/outputDelta: delta field (not output!)
        - reasoning deltas: delta field
        - fileChange/outputDelta: delta field (not output!)
        """
        from codex_adapter.models import (
            AgentMessageDeltaData,
            CommandExecutionOutputDeltaData,
            FileChangeOutputDeltaData,
            ReasoningSummaryTextDeltaData,
            ReasoningTextDeltaData,
        )

        if not self.is_delta():
            return ""

        # Type-safe extraction based on actual type
        if isinstance(
            self.data,
            AgentMessageDeltaData
            | ReasoningTextDeltaData
            | ReasoningSummaryTextDeltaData
            | CommandExecutionOutputDeltaData
            | FileChangeOutputDeltaData,
        ):
            return self.data.delta

        return ""
