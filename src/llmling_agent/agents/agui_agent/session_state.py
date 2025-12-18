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


@dataclass
class AGUISessionState:
    """Track state for an active AG-UI session.

    Maintains session state throughout an AG-UI run,
    including shared state from the remote agent.
    """

    thread_id: str
    """Thread ID for this session."""

    run_id: str | None = None
    """Current run ID."""

    is_complete: bool = False
    """Whether the current run is complete."""

    error: str | None = None
    """Error message if run failed."""

    # Extended state tracking
    state: dict[str, Any] = field(default_factory=dict)
    """Shared state from StateSnapshot/StateDelta events."""

    current_step: str | None = None
    """Current step name if in a step."""

    # Reconnection support
    processed_message_ids: set[str] = field(default_factory=set)
    """Message IDs already processed (for deduplication on reconnect)."""

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

    def clear(self) -> None:
        """Clear session state for new run."""
        self.state.clear()
        self.processed_message_ids.clear()
        self.is_complete = False
        self.current_step = None
        self.error = None
        self.run_id = str(uuid4())
