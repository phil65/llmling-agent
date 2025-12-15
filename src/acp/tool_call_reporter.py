"""Stateful tool call progress reporter for ACP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from acp.schema.tool_call import ToolCallLocation


if TYPE_CHECKING:
    from collections.abc import Sequence
    import types

    from acp.notifications import ACPNotifications
    from acp.schema.tool_call import (
        ToolCallContent,
        ToolCallKind,
        ToolCallStatus,
    )


class ToolCallReporter:
    """Stateful reporter for tool call progress.

    Maintains the current state of a tool call and sends updates
    when fields are modified. This avoids having to repeat unchanged
    fields on every progress update.

    Example:
        ```python
        reporter = await notifications.create_tool_reporter(
            tool_call_id="abc123",
            title="Reading file",
            kind="read",
        )
        await reporter.update(status="in_progress", message="Opening file...")
        await reporter.update(progress=0.5)
        await reporter.update(status="completed", message="Done!")
        ```
    """

    def __init__(
        self,
        notifications: ACPNotifications,
        tool_call_id: str,
        title: str,
        *,
        kind: ToolCallKind | None = None,
        status: ToolCallStatus = "pending",
        locations: Sequence[ToolCallLocation] | None = None,
        content: Sequence[ToolCallContent] | None = None,
        raw_input: Any | None = None,
        raw_output: Any | None = None,
    ) -> None:
        """Initialize the reporter with initial state.

        Args:
            notifications: The ACPNotifications instance to send updates through
            tool_call_id: Unique identifier for this tool call
            title: Human-readable title describing the tool action
            kind: Category of tool being invoked
            status: Initial execution status
            locations: File locations affected by this tool call
            content: Content produced by the tool call
            raw_input: Raw input parameters sent to the tool
            raw_output: Raw output returned by the tool
        """
        self._notifications = notifications
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind: ToolCallKind | None = kind
        self.status: ToolCallStatus = status
        self.locations: list[ToolCallLocation] = list(locations) if locations else []
        self.content: list[ToolCallContent] = list(content) if content else []
        self.raw_input = raw_input
        self.raw_output = raw_output
        self._started = False

    async def start(self) -> None:
        """Send the initial tool_call notification.

        This is called automatically if you use `create_tool_reporter`,
        but can be called manually if constructing the reporter directly.
        """
        if self._started:
            return

        await self._notifications.tool_call_start(
            tool_call_id=self.tool_call_id,
            title=self.title,
            kind=self.kind,
            locations=self.locations or None,
            content=self.content or None,
            raw_input=self.raw_input,
        )
        self._started = True

    async def update(
        self,
        *,
        title: str | None = None,
        status: ToolCallStatus | None = None,
        kind: ToolCallKind | None = None,
        message: str | None = None,
        raw_output: Any | None = None,
        locations: Sequence[ToolCallLocation | str] | None = None,
        content: Sequence[ToolCallContent | str] | None = None,
        replace: bool = False,
    ) -> None:
        """Update one or more fields and send the updated state.

        Args:
            title: Update the human-readable title
            status: Update the execution status
            kind: Update the tool kind
            message: Convenience for adding a text content block
            raw_output: Update the raw output
            locations: Locations to add or replace (path strings or ToolCallLocation)
            content: Content blocks to add or replace
            replace: If True, replace all content/locations; if False, append
        """
        if not self._started:
            await self.start()

        # Update stored state
        if title is not None:
            self.title = title
        if status is not None:
            self.status = status
        if kind is not None:
            self.kind = kind
        if raw_output is not None:
            self.raw_output = raw_output

        # Handle locations
        if locations is not None:
            normalized = [
                ToolCallLocation(path=loc) if isinstance(loc, str) else loc for loc in locations
            ]
            if replace:
                self.locations = normalized
            else:
                self.locations.extend(normalized)

        # Handle content (strings will be converted by tool_call_progress)
        if content is not None:
            content_list: list[ToolCallContent | str] = list(content)
            if replace:
                self.content = content_list  # type: ignore[assignment]
            else:
                self.content.extend(content_list)  # type: ignore[arg-type]

        if message is not None:
            self.content.append(message)  # type: ignore[arg-type]

        await self._notifications.tool_call_progress(
            tool_call_id=self.tool_call_id,
            status=self.status,
            title=self.title,
            kind=self.kind,
            locations=self.locations or None,
            content=self.content or None,
            raw_output=self.raw_output,
        )

    async def complete(self, *, message: str | None = None, raw_output: Any | None = None) -> None:
        """Mark the tool call as completed.

        Args:
            message: Optional final message to add
            raw_output: Optional final output
        """
        await self.update(status="completed", message=message, raw_output=raw_output)

    async def fail(self, *, error: str | None = None, raw_output: Any | None = None) -> None:
        """Mark the tool call as failed.

        Args:
            error: Optional error message to add
            raw_output: Optional output (e.g., error details)
        """
        await self.update(status="failed", message=error, raw_output=raw_output)

    async def __aenter__(self) -> Self:
        """Enter async context, starting the tool call."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit async context, marking complete or failed based on exception."""
        if exc_type is not None:
            await self.fail(error=str(exc_val) if exc_val else "Unknown error")
        elif self.status not in ("completed", "failed"):
            await self.complete()
