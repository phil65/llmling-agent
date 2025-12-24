"""Stateful tool call tracker for ACP.

This module provides a simple state tracker for tool calls that sends
only changed fields in updates, following the ACP protocol spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp.notifications import ACPNotifications
    from acp.schema.tool_call import ToolCallContent, ToolCallKind, ToolCallLocation, ToolCallStatus


class ToolCallState:
    """Tracks tool call state and sends delta updates.

    Instead of accumulating all state and sending everything on each update,
    this class tracks whether we've started and sends only the fields that
    are explicitly provided in each update call.

    The ACP protocol spec states:
    "All fields except toolCallId are optional in updates.
     Only the fields being changed need to be included."

    Example flow:
        1. Tool call starts → state.start() sends initial notification
        2. Progress event → state.update(title="Running...") sends just title
        3. More progress → state.update(content=[...]) sends just content
        4. Tool completes → state.complete(raw_output=result) sends status + output
    """

    def __init__(
        self,
        notifications: ACPNotifications,
        tool_call_id: str,
        tool_name: str,
        title: str,
        kind: ToolCallKind,
        raw_input: dict[str, Any],
    ) -> None:
        """Initialize tool call state.

        Args:
            notifications: ACPNotifications instance for sending updates
            tool_call_id: Unique identifier for this tool call
            tool_name: Name of the tool being called
            title: Initial human-readable title
            kind: Category of tool (read, edit, execute, etc.)
            raw_input: Input parameters passed to the tool
        """
        self._notifications = notifications
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self._initial_title = title
        self._initial_kind: ToolCallKind = kind
        self._raw_input = raw_input
        self._started = False
        self._has_content = False  # Track if content has been sent

    @property
    def has_content(self) -> bool:
        """Whether content has been sent for this tool call."""
        return self._has_content

    async def start(self) -> None:
        """Send initial tool_call notification.

        This creates the tool call entry in the client UI. Subsequent calls
        to update() will send tool_call_update notifications with only
        the changed fields.
        """
        if self._started:
            return

        await self._notifications.tool_call_start(
            tool_call_id=self.tool_call_id,
            title=self._initial_title,
            kind=self._initial_kind,
            raw_input=self._raw_input,
        )
        self._started = True

    async def update(
        self,
        *,
        title: str | None = None,
        status: ToolCallStatus | None = None,
        kind: ToolCallKind | None = None,
        content: Sequence[ToolCallContent] | None = None,
        locations: Sequence[ToolCallLocation | str] | None = None,
        raw_output: Any = None,
    ) -> None:
        """Send an update with only the provided fields.

        Args:
            title: Update the human-readable title
            status: Update execution status
            kind: Update tool kind
            content: Content items (terminals, diffs, text) - replaces previous
            locations: File locations - replaces previous
            raw_output: Update raw output data

        Note:
            Only fields that are truthy (not None/empty) will be included
            in the notification. This follows the ACP spec which states
            that only changed fields need to be sent.
        """
        from acp.schema.tool_call import ToolCallLocation

        if not self._started:
            await self.start()
        # Build kwargs with only the provided fields
        kwargs: dict[str, Any] = {}
        if title is not None:
            kwargs["title"] = title
        if status is not None:
            kwargs["status"] = status
        if kind is not None:
            kwargs["kind"] = kind
        if content:
            kwargs["content"] = content
            self._has_content = True
        if locations:
            # Normalize string paths to ToolCallLocation
            kwargs["locations"] = [
                ToolCallLocation(path=loc) if isinstance(loc, str) else loc for loc in locations
            ]
        if raw_output is not None:
            kwargs["raw_output"] = raw_output

        # Only send if there's something to update
        if kwargs:
            await self._notifications.tool_call_update(
                tool_call_id=self.tool_call_id,
                **kwargs,
            )

    async def complete(
        self,
        raw_output: Any = None,
        *,
        content: Sequence[ToolCallContent] | None = None,
        title: str | None = None,
    ) -> None:
        """Mark tool call as completed.

        Args:
            raw_output: Final output data
            content: Optional final content
            title: Optional final title
        """
        await self.update(
            status="completed",
            raw_output=raw_output,
            content=content,
            title=title,
        )

    async def fail(self, error: str | None = None, *, raw_output: Any = None) -> None:
        """Mark tool call as failed.

        Args:
            error: Error message to display
            raw_output: Optional error details
        """
        from acp.schema import ContentToolCallContent

        content = None
        if error:
            error_content = ContentToolCallContent.text(text=f"Error: {error}")
            content = [error_content]

        await self.update(status="failed", content=content, raw_output=raw_output)
