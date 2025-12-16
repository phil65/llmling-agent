"""ACP Agent - MessageNode wrapping an external ACP subprocess."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class AGUISessionState:
    """Track state for an active AG-UI session."""

    thread_id: str
    """Thread ID for this session."""

    run_id: str | None = None
    """Current run ID."""

    text_chunks: list[str] = field(default_factory=list)
    """Accumulated text chunks."""

    thought_chunks: list[str] = field(default_factory=list)
    """Accumulated thought chunks."""

    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Active tool calls by ID."""

    is_complete: bool = False
    """Whether the current run is complete."""

    error: str | None = None
    """Error message if run failed."""

    def clear(self) -> None:
        """Clear session state."""
        self.text_chunks.clear()
        self.thought_chunks.clear()
        self.tool_calls.clear()
        self.is_complete = False
        self.error = None
        self.run_id = str(uuid4())
