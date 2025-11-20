"""Event emitter delegate for AgentContext with fluent API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.agent.events import LocationContentItem


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext
    from llmling_agent.agent.events import RichAgentStreamEvent, ToolCallContentItem
    from llmling_agent.resource_providers.plan_provider import PlanEntry
    from llmling_agent.tools.base import ToolKind


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

        event = PlanUpdateEvent(entries=entries.copy(), tool_call_id=self._context.tool_call_id)
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

    async def process_started(
        self,
        process_id: str,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Emit process start event.

        Args:
            process_id: Unique process identifier
            command: Command being executed
            args: Command arguments
            cwd: Working directory
            env: Environment variables
            output_limit: Maximum bytes of output to retain
            success: Whether the process started successfully
            error: Error message if start failed
        """
        from llmling_agent.agent.events import ProcessStartEvent

        event = ProcessStartEvent(
            process_id=process_id,
            command=command,
            args=args or [],
            cwd=cwd,
            env=env or {},
            output_limit=output_limit,
            success=success,
            error=error,
            tool_call_id=self._context.tool_call_id,
        )
        await self._context.agent._event_queue.put(event)

    async def process_output(
        self,
        process_id: str,
        output: str,
        stdout: str | None = None,
        stderr: str | None = None,
        truncated: bool = False,
    ) -> None:
        """Emit process output event.

        Args:
            process_id: Process identifier
            output: Process output (stdout/stderr combined)
            stdout: Standard output (if separated)
            stderr: Standard error (if separated)
            truncated: Whether output was truncated due to limits
        """
        from llmling_agent.agent.events import ProcessOutputEvent

        event = ProcessOutputEvent(
            process_id=process_id,
            output=output,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
            tool_call_id=self._context.tool_call_id,
        )
        await self._context.agent._event_queue.put(event)

    async def process_exit(
        self,
        process_id: str,
        exit_code: int,
        final_output: str | None = None,
        truncated: bool = False,
    ) -> None:
        """Emit process exit event.

        Args:
            process_id: Process identifier
            exit_code: Process exit code
            final_output: Final process output
            truncated: Whether output was truncated due to limits
        """
        from llmling_agent.agent.events import ProcessExitEvent

        event = ProcessExitEvent(
            process_id=process_id,
            exit_code=exit_code,
            success=exit_code == 0,
            final_output=final_output,
            truncated=truncated,
            tool_call_id=self._context.tool_call_id,
        )
        await self._context.agent._event_queue.put(event)

    async def process_killed(
        self,
        process_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Emit process kill event.

        Args:
            process_id: Process identifier
            success: Whether the process was successfully killed
            error: Error message if kill failed
        """
        from llmling_agent.agent.events import ProcessKillEvent

        event = ProcessKillEvent(
            process_id=process_id,
            success=success,
            error=error,
            tool_call_id=self._context.tool_call_id,
        )
        await self._context.agent._event_queue.put(event)

    async def process_released(
        self,
        process_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Emit process release event.

        Args:
            process_id: Process identifier
            success: Whether resources were successfully released
            error: Error message if release failed
        """
        from llmling_agent.agent.events import ProcessReleaseEvent

        event = ProcessReleaseEvent(
            process_id=process_id,
            success=success,
            error=error,
            tool_call_id=self._context.tool_call_id,
        )
        await self._context.agent._event_queue.put(event)

    async def tool_call_start(
        self,
        title: str,
        kind: ToolKind = "other",
        content: list[ToolCallContentItem] | None = None,
        locations: list[str | LocationContentItem] | None = None,
    ) -> None:
        """Emit tool call start event with rich ACP metadata.

        Args:
            title: Human-readable title describing what the tool is doing
            kind: Tool kind (read, edit, delete, move, search, execute, think, fetch, other)
            content: Content produced by the tool call (terminals, diffs, text)
            locations: File paths or LocationContentItem objects affected by this tool call
        """
        from llmling_agent.agent.events import ToolCallStartEvent

        # Convert string paths to LocationContentItem objects
        location_items: list[LocationContentItem] = []
        if locations:
            for loc in locations:
                if isinstance(loc, str):
                    location_items.append(LocationContentItem(path=loc))
                else:
                    location_items.append(loc)

        event = ToolCallStartEvent(
            tool_call_id=self._context.tool_call_id or "",
            tool_name=self._context.tool_name or "",
            title=title,
            kind=kind,
            content=content or [],
            locations=location_items,
            raw_input=self._context.tool_input.copy(),
        )

        event = ToolCallStartEvent(
            tool_call_id=self._context.tool_call_id or "",
            tool_name=self._context.tool_name or "",
            title=title,
            kind=kind,
            locations=location_items,
            raw_input=self._context.tool_input.copy(),
        )
        await self._context.agent._event_queue.put(event)

    async def emit_event(self, event: RichAgentStreamEvent[Any]) -> None:
        """Emit a typed event into the agent's event stream.

        Args:
            event: The event instance (PlanUpdateEvent, FileOperationEvent, etc.)
        """
        await self._context.agent._event_queue.put(event)
