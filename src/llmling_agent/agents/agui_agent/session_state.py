"""AG-UI session state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class AGUICheckpoint:
    """Checkpoint for resuming AG-UI sessions after disconnection."""

    thread_id: str
    """Thread ID for this session."""

    run_id: str | None
    """Run ID at checkpoint time."""

    state: dict[str, Any]
    """Shared state snapshot."""

    message_ids: list[str]
    """Processed message IDs for deduplication."""

    text_content: str
    """Accumulated text content at checkpoint."""

    thinking_content: str
    """Accumulated thinking content at checkpoint."""


@dataclass
class AGUISessionState:
    """Track state for an active AG-UI session.

    Maintains accumulated content and state throughout an AG-UI run,
    including text messages, thinking/reasoning content, tool calls,
    and shared state from the remote agent.
    """

    thread_id: str
    """Thread ID for this session."""

    run_id: str | None = None
    """Current run ID."""

    text_chunks: list[str] = field(default_factory=list)
    """Accumulated text chunks from assistant messages."""

    thought_chunks: list[str] = field(default_factory=list)
    """Accumulated thinking/reasoning chunks."""

    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Active tool calls by ID: {tool_call_id: {name, args_buffer}}."""

    is_complete: bool = False
    """Whether the current run is complete."""

    error: str | None = None
    """Error message if run failed."""

    # Extended state tracking
    state: dict[str, Any] = field(default_factory=dict)
    """Shared state from StateSnapshot/StateDelta events."""

    is_thinking: bool = False
    """Whether we're currently in a thinking block."""

    current_step: str | None = None
    """Current step name if in a step."""

    # Reconnection support
    processed_message_ids: set[str] = field(default_factory=set)
    """Message IDs already processed (for deduplication on reconnect)."""

    @property
    def text_content(self) -> str:
        """Get accumulated text content."""
        return "".join(self.text_chunks)

    @property
    def thinking_content(self) -> str:
        """Get accumulated thinking/reasoning content."""
        return "".join(self.thought_chunks)

    def add_text(self, delta: str) -> None:
        """Add text content delta."""
        self.text_chunks.append(delta)

    def add_thinking(self, delta: str) -> None:
        """Add thinking content delta."""
        self.thought_chunks.append(delta)

    def start_thinking(self) -> None:
        """Mark start of thinking block."""
        self.is_thinking = True

    def end_thinking(self) -> None:
        """Mark end of thinking block."""
        self.is_thinking = False

    def mark_message_processed(self, message_id: str) -> None:
        """Mark a message ID as processed."""
        self.processed_message_ids.add(message_id)

    def is_message_processed(self, message_id: str) -> bool:
        """Check if a message has already been processed."""
        return message_id in self.processed_message_ids

    def apply_state_snapshot(self, snapshot: Any) -> None:
        """Apply a state snapshot, replacing current state."""
        if isinstance(snapshot, dict):
            self.state = snapshot.copy()
        else:
            self.state = {"value": snapshot}

    def apply_state_delta(self, delta: list[dict[str, Any]]) -> None:
        """Apply JSON Patch operations to state.

        Args:
            delta: List of RFC 6902 JSON Patch operations
        """
        try:
            import importlib

            jsonpatch = importlib.import_module("jsonpatch")
            self.state = jsonpatch.apply_patch(self.state, delta)
        except (ImportError, ModuleNotFoundError):
            # Fallback: just store delta info if jsonpatch not available
            self.state["_pending_delta"] = delta
        except Exception:  # noqa: BLE001
            # Log but don't fail on patch errors
            pass

    def create_checkpoint(self) -> AGUICheckpoint:
        """Create a checkpoint for potential session resume.

        Returns:
            Checkpoint containing current state for reconnection
        """
        return AGUICheckpoint(
            thread_id=self.thread_id,
            run_id=self.run_id,
            state=self.state.copy(),
            message_ids=list(self.processed_message_ids),
            text_content=self.text_content,
            thinking_content=self.thinking_content,
        )

    def restore_from_checkpoint(self, checkpoint: AGUICheckpoint) -> None:
        """Restore session state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        self.thread_id = checkpoint.thread_id
        self.run_id = checkpoint.run_id
        self.state = checkpoint.state.copy()
        self.processed_message_ids = set(checkpoint.message_ids)
        # Restore accumulated content
        self.text_chunks = [checkpoint.text_content] if checkpoint.text_content else []
        self.thought_chunks = [checkpoint.thinking_content] if checkpoint.thinking_content else []

    def clear(self) -> None:
        """Clear session state for new run."""
        self.text_chunks.clear()
        self.thought_chunks.clear()
        self.tool_calls.clear()
        self.state.clear()
        self.processed_message_ids.clear()
        self.is_complete = False
        self.is_thinking = False
        self.current_step = None
        self.error = None
        self.run_id = str(uuid4())
