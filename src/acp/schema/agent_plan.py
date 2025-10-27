from __future__ import annotations

from typing import Any, Literal

from acp.base import Schema


PlanEntryPriority = Literal["high", "medium", "low"]
PlanEntryStatus = Literal["pending", "in_progress", "completed"]


class PlanEntry(Schema):
    """A single entry in the execution plan.

    Represents a task or goal that the assistant intends to accomplish
    as part of fulfilling the user's request.
    See protocol docs: [Plan Entries](https://agentclientprotocol.com/protocol/agent-plan#plan-entries)
    """

    field_meta: Any | None = None
    """Extension point for implementations."""

    content: str
    """Human-readable description of what this task aims to accomplish."""

    priority: PlanEntryPriority
    """The relative importance of this task.

    Used to indicate which tasks are most critical to the overall goal.
    """

    status: PlanEntryStatus
    """Current execution status of this task."""
