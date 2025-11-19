"""Event emitter delegate for AgentContext with fluent API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext
    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.resource_providers.plan_provider import PlanEntry


class AgentEventEmitter:
    """Event emitter delegate that automatically injects context.

    Provides a fluent, developer-friendly API for emitting domain events
    with context (tool_call_id, etc.) automatically injected.
    """

    def __init__(self, context: AgentContext) -> None:
        """Initialize event emitter with agent context.

        Args:
            context: Agent context to extract metadata from
        """
        self._context = context

    async def file_operation(
        self,
        operation: Literal["read", "write", "delete", "list", "edit"],
        path: str,
        success: bool,
        error: str | None = None,
        size: int | None = None,
        title: str | None = None,
        kind: str | None = None,
        locations: list[str] | None = None,
        raw_output: Any | None = None,
    ) -> None:
        """Emit file operation event with rich metadata.

        Args:
            operation: The filesystem operation performed
            path: The file/directory path that was operated on
            success: Whether the operation completed successfully
            error: Error message if operation failed
            size: Size of file in bytes (for successful operations)
            title: Display title for the operation
            kind: Tool operation kind (edit, read, write, etc.)
            locations: File paths affected by the operation
            raw_output: Tool result data for failed operations
        """
        from llmling_agent.agent.events import FileOperationEvent

        event = FileOperationEvent(
            operation=operation,
            path=path,
            success=success,
            error=error,
            size=size,
            tool_call_id=self._context.tool_call_id,
            title=title,
            kind=kind or operation,  # Default kind to operation
            locations=locations or [path],  # Default to main path
            raw_input=self._context.tool_input.copy(),  # Auto-inject from context
            raw_output=raw_output,
        )
        await self._context.agent._event_queue.put(event)

    async def file_edit_progress(
        self,
        path: str,
        old_text: str,
        new_text: str,
        status: Literal["in_progress", "completed", "failed"],
        changed_lines: list[int] | None = None,
    ) -> None:
        """Emit file edit progress event with diff information.

        Args:
            path: The file path being edited
            old_text: Original file content
            new_text: New file content
            status: Current status of the edit operation
            changed_lines: Line numbers that were changed
        """
        from llmling_agent.agent.events import FileEditProgressEvent

        event = FileEditProgressEvent(
            path=path,
            old_text=old_text,
            new_text=new_text,
            status=status,
            changed_lines=changed_lines or [],
            tool_call_id=self._context.tool_call_id,
        )
        await self._context.agent._event_queue.put(event)

    async def plan_updated(self, entries: list[PlanEntry]) -> None:
        """Emit plan update event.

        Args:
            entries: Current plan entries
        """
        from llmling_agent.agent.events import PlanUpdateEvent

        event = PlanUpdateEvent(
            entries=entries.copy(), tool_call_id=self._context.tool_call_id
        )
        await self.emit_event(event)

    async def progress(self, progress: float, total: float | None, message: str) -> None:
        """Emit progress event (delegates to existing method).

        Args:
            progress: Current progress value
            total: Total progress value
            message: Progress message
        """
        await self._context.report_progress(progress, total, message)

    async def custom(
        self, event_data: Any, event_type: str = "custom", source: str | None = None
    ) -> None:
        """Emit custom event.

        Args:
            event_data: The custom event data of any type
            event_type: Type identifier for the custom event
            source: Optional source identifier
        """
        from llmling_agent.agent.events import CustomEvent

        custom_event = CustomEvent(
            event_data=event_data,
            event_type=event_type,
            source=source or self._context.tool_name,
        )
        await self._context.agent._event_queue.put(custom_event)

    async def emit_event(self, event: RichAgentStreamEvent) -> None:
        """Emit a typed event into the agent's event stream.

        Args:
            event: The event instance (PlanUpdateEvent, FileOperationEvent, etc.)
        """
        await self._context.agent._event_queue.put(event)
