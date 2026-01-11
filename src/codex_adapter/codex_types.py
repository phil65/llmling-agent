"""Codex data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Type aliases for Codex types
ModelProvider = Literal["openai", "anthropic", "google", "mistral"]
TurnStatus = Literal["pending", "running", "completed", "error", "interrupted"]
ItemType = Literal[
    "reasoning",
    "agent_message",
    "command_execution",
    "user_message",
    "file_change",
    "mcp_tool_call",
]
ItemStatus = Literal["pending", "running", "completed", "error"]


@dataclass
class CodexThread:
    """Represents a Codex conversation thread."""

    id: str
    preview: str = ""
    model_provider: ModelProvider = "openai"
    created_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexTurn:
    """Represents a turn in a Codex conversation."""

    id: str
    thread_id: str
    status: TurnStatus = "pending"
    items: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class CodexItem:
    """Represents an item (message, tool call, etc.) in a turn."""

    id: str
    type: ItemType
    content: str = ""
    status: ItemStatus = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)
