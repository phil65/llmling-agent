"""Codex event types for streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


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
        data: The notification params (event-specific payload)
        raw: The full JSON-RPC notification message
    """

    event_type: str
    data: dict[str, Any]
    raw: dict[str, Any]

    @classmethod
    def from_notification(cls, method: str, params: dict[str, Any] | None) -> CodexEvent:
        """Create event from JSON-RPC notification."""
        return cls(
            event_type=method,
            data=params or {},
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

        Both agentMessage and commandExecution deltas use 'delta' field.
        """
        if "delta" in self.event_type.lower():
            delta = self.data.get("delta", "")
            return str(delta) if delta else ""
        return ""
