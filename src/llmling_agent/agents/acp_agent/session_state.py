"""ACP Agent - MessageNode wrapping an external ACP subprocess."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

from llmling_agent.log import get_logger


logger = get_logger(__name__)

PROTOCOL_VERSION = 1


@dataclass
class ACPSessionState:
    """Tracks state of an ACP session."""

    session_id: str
    """The session ID from the ACP server."""

    text_chunks: list[str] = dataclass_field(default_factory=list)
    """Accumulated text chunks."""

    thought_chunks: list[str] = dataclass_field(default_factory=list)
    """Accumulated thought/reasoning chunks."""

    tool_calls: list[dict[str, Any]] = dataclass_field(default_factory=list)
    """Tool call records."""

    events: list[Any] = dataclass_field(default_factory=list)
    """Queue of native events converted from ACP updates."""

    is_complete: bool = False
    """Whether the prompt processing is complete."""

    stop_reason: str | None = None
    """Reason processing stopped."""

    current_model_id: str | None = None
    """Current model ID from session state."""

    def clear(self) -> None:
        self.text_chunks.clear()
        self.thought_chunks.clear()
        self.tool_calls.clear()
        self.events.clear()
        self.is_complete = False
        self.stop_reason = None
        self.current_model_id = None
