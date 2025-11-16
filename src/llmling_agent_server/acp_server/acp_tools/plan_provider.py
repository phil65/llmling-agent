"""ACP plan provider for agent planning and task management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp import PlanEntry, PlanEntryPriority, PlanEntryStatus  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent_server.acp_server.session import ACPSession


class ACPPlanProvider(ResourceProvider):
    """Provides ACP plan-related tools for agent planning and task management.

    This provider creates session-aware tools for managing agent plans and tasks
    via the ACP client. All tools have the session ID baked in at creation time,
    eliminating the need for parameter injection.
    """

    def __init__(self, session: ACPSession) -> None:
        """Initialize plan provider.

        Args:
            session: The ACP session instance
        """
        super().__init__(name=f"acp_plan_{session.session_id}")
        self.session = session
        self.session_id = session.session_id
        self._current_plan: list[PlanEntry] = []

    async def get_tools(self) -> list[Tool]:
        """Get plan management tools."""
        return [
            Tool.from_callable(self.add_plan_entry, source="planning", category="other"),
            Tool.from_callable(
                self.update_plan_entry, source="planning", category="edit"
            ),
            Tool.from_callable(
                self.remove_plan_entry, source="planning", category="delete"
            ),
        ]

    async def add_plan_entry(
        self,
        content: str,
        priority: PlanEntryPriority = "medium",
        index: int | None = None,
    ) -> str:
        """Add a new plan entry.

        Args:
            content: Description of what this task aims to accomplish
            priority: Relative importance (high/medium/low)
            index: Optional position to insert at (default: append to end)

        Returns:
            Success message indicating entry was added
        """
        entry = PlanEntry(content=content, priority=priority, status="pending")
        if index is None:
            self._current_plan.append(entry)
            entry_index = len(self._current_plan) - 1
        else:
            if index < 0 or index > len(self._current_plan):
                return f"Error: Index {index} out of range (0-{len(self._current_plan)})"
            self._current_plan.insert(index, entry)
            entry_index = index

        await self._send_plan_update()

        return f"Added plan entry at index {entry_index}: {content!r} ({priority=!r})"

    async def update_plan_entry(
        self,
        index: int,
        content: str | None = None,
        status: PlanEntryStatus | None = None,
        priority: PlanEntryPriority | None = None,
    ) -> str:
        """Update an existing plan entry.

        Args:
            index: Position of entry to update (0-based)
            content: New task description
            status: New execution status
            priority: New priority level

        Returns:
            Success message indicating what was updated
        """
        if index < 0 or index >= len(self._current_plan):
            return f"Error: Index {index} out of range (0-{len(self._current_plan) - 1})"

        entry = self._current_plan[index]
        updates = []

        if content is not None:
            entry.content = content
            updates.append(f"content to {content!r}")

        if status is not None:
            entry.status = status
            updates.append(f"status to {status!r}")

        if priority is not None:
            entry.priority = priority
            updates.append(f"priority to {priority!r}")

        if not updates:
            return "No changes specified"

        await self._send_plan_update()
        return f"Updated entry {index}: {', '.join(updates)}"

    async def remove_plan_entry(self, index: int) -> str:
        """Remove a plan entry.

        Args:
            index: Position of entry to remove (0-based)

        Returns:
            Success message indicating entry was removed
        """
        if index < 0 or index >= len(self._current_plan):
            return f"Error: Index {index} out of range (0-{len(self._current_plan) - 1})"
        removed_entry = self._current_plan.pop(index)
        await self._send_plan_update()
        if self._current_plan:
            return (
                f"Removed entry {index}: {removed_entry.content!r}, "
                f"remaining entries reindexed"
            )
        return f"Removed entry {index}: {removed_entry.content!r}, plan is now empty"

    async def _send_plan_update(self) -> None:
        """Send current plan state via session update."""
        if not self._current_plan:  # Don't send empty plans
            return
        await self.session.notifications.update_plan(self._current_plan)
