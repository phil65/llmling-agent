"""Plan provider for agent planning and task management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool


# Plan entry types - domain models independent of ACP
PlanEntryPriority = Literal["high", "medium", "low"]
PlanEntryStatus = Literal["pending", "in_progress", "completed"]


@dataclass(kw_only=True)
class PlanEntry:
    """A single entry in the execution plan.

    Represents a task or goal that the assistant intends to accomplish
    as part of fulfilling the user's request.
    """

    content: str
    """Human-readable description of what this task aims to accomplish."""

    priority: PlanEntryPriority
    """The relative importance of this task."""

    status: PlanEntryStatus
    """Current execution status of this task."""


@dataclass(kw_only=True)
class PlanUpdateEvent:
    """Event indicating plan state has changed."""

    entries: list[PlanEntry]
    """Current plan entries."""
    event_kind: Literal["plan_update"] = "plan_update"
    """Event type identifier."""


class PlanProvider(ResourceProvider):
    """Provides plan-related tools for agent planning and task management.

    This provider creates tools for managing agent plans and tasks,
    emitting domain events that can be handled by protocol adapters.
    """

    def __init__(self) -> None:
        """Initialize plan provider."""
        super().__init__(name="plan")
        self._current_plan: list[PlanEntry] = []

    async def get_tools(self) -> list[Tool]:
        """Get plan management tools."""
        return [
            self.create_tool(self.get_plan, category="read"),
            self.create_tool(self.add_plan_entry, category="other"),
            self.create_tool(self.update_plan_entry, category="edit"),
            self.create_tool(self.remove_plan_entry, category="delete"),
        ]

    async def get_plan(self, agent_ctx: AgentContext) -> str:
        """Get the current plan formatted as markdown.

        Args:
            agent_ctx: Agent execution context

        Returns:
            Markdown-formatted plan with all entries and their status
        """
        if not self._current_plan:
            return "## Plan\n\n*No plan entries yet.*"

        lines = ["## Plan", ""]
        status_icons = {
            "pending": "⬚",
            "in_progress": "◐",
            "completed": "✓",
        }
        priority_labels = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢",
        }
        for i, entry in enumerate(self._current_plan):
            icon = status_icons.get(entry.status, "?")
            priority = priority_labels.get(entry.priority, "")
            lines.append(f"{i}. {icon} {priority} {entry.content} *({entry.status})*")

        return "\n".join(lines)

    async def add_plan_entry(
        self,
        agent_ctx: AgentContext,
        content: str,
        priority: PlanEntryPriority = "medium",
        index: int | None = None,
    ) -> str:
        """Add a new plan entry.

        Args:
            agent_ctx: Agent execution context
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

        await self._emit_plan_update(agent_ctx)

        return f"Added plan entry at index {entry_index}: {content!r} (priority={priority!r})"

    async def update_plan_entry(
        self,
        agent_ctx: AgentContext,
        index: int,
        content: str | None = None,
        status: PlanEntryStatus | None = None,
        priority: PlanEntryPriority | None = None,
    ) -> str:
        """Update an existing plan entry.

        Args:
            agent_ctx: Agent execution context
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

        await self._emit_plan_update(agent_ctx)
        return f"Updated entry {index}: {', '.join(updates)}"

    async def remove_plan_entry(self, agent_ctx: AgentContext, index: int) -> str:
        """Remove a plan entry.

        Args:
            agent_ctx: Agent execution context
            index: Position of entry to remove (0-based)

        Returns:
            Success message indicating entry was removed
        """
        if index < 0 or index >= len(self._current_plan):
            return f"Error: Index {index} out of range (0-{len(self._current_plan) - 1})"
        removed_entry = self._current_plan.pop(index)
        await self._emit_plan_update(agent_ctx)
        if self._current_plan:
            return f"Removed entry {index}: {removed_entry.content!r}, remaining entries reindexed"
        return f"Removed entry {index}: {removed_entry.content!r}, plan is now empty"

    async def _emit_plan_update(self, agent_ctx: AgentContext) -> None:
        """Emit plan update event."""
        await agent_ctx.events.plan_updated(self._current_plan)
