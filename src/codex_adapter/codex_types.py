"""Codex data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodexThread:
    """Represents a Codex conversation thread."""

    id: str
    preview: str = ""
    model_provider: str = "openai"
    created_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexTurn:
    """Represents a turn in a Codex conversation."""

    id: str
    thread_id: str
    status: str = "pending"
    items: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class CodexItem:
    """Represents an item (message, tool call, etc.) in a turn."""

    id: str
    type: str  # "reasoning", "agent_message", "command_execution", "user_message", etc.
    content: str = ""
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)
