"""Stateful tool call accumulator for ACP.

This module provides a unified state management approach for tool calls,
ensuring that rich progress information (diffs, terminals, locations) is
accumulated rather than overwritten by subsequent notifications.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from acp.schema import ContentToolCallContent, TerminalToolCallContent
from acp.schema.tool_call import ToolCallLocation


if TYPE_CHECKING:
    from acp.notifications import ACPNotifications
    from acp.schema.tool_call import ToolCallContent, ToolCallKind, ToolCallStatus


class ToolCallState:
    """Accumulates tool call state across the execution lifecycle.

    Instead of each event handler independently sending notifications that
    overwrite previous state, all updates go through this accumulator which
    preserves content, locations, and other rich data across the tool call
    lifecycle.

    Example flow:
        1. Tool call starts → state created with generated title
        2. ToolCallProgressEvent → state.update(title="Running: ...", add_content=terminal)
        3. ToolCallProgressEvent → state.update(add_content=diff, add_locations=path)
        4. Tool completes → state.complete(raw_output=result)

    All accumulated content and locations are preserved in each notification.
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
            title: Initial human-readable title (can be updated later)
            kind: Category of tool (read, edit, execute, etc.)
            raw_input: Input parameters passed to the tool
        """
        self._notifications = notifications
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.title = title
        self.kind: ToolCallKind = kind
        self.status: ToolCallStatus = "pending"
        self.content: list[ToolCallContent] = []
        self.locations: list[ToolCallLocation] = []
        self.raw_input = raw_input
        self.raw_output: Any = None
        self._started = False

    async def start(self) -> None:
        """Send initial tool_call notification.

        This creates the tool call entry in the client UI. Subsequent calls
        to update() will send tool_call_update notifications.
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
        add_content: ToolCallContent | Sequence[ToolCallContent] | None = None,
        replace_content: bool = False,
        add_location: ToolCallLocation | str | None = None,
        add_locations: Sequence[ToolCallLocation | str] | None = None,
        raw_output: Any = None,
    ) -> None:
        """Update state and send notification with ALL accumulated data.

        Unlike direct notification calls, this method preserves all previously
        accumulated content and locations, only adding new data.

        Args:
            title: Override the human-readable title
            status: Update execution status
            kind: Update tool kind
            add_content: Content to append or replace (terminals, diffs, text)
            replace_content: If True, replace all content instead of appending
            add_location: Single location to append
            add_locations: Multiple locations to append
            raw_output: Update raw output data
        """
        if not self._started:
            await self.start()

        # Update scalar fields
        if title is not None:
            self.title = title
        if status is not None:
            self.status = status
        if kind is not None:
            self.kind = kind
        if raw_output is not None:
            self.raw_output = raw_output

        # Handle content: replace or accumulate
        if add_content is not None:
            if replace_content:
                # Replace all existing content
                if isinstance(add_content, Sequence):
                    self.content = list(add_content)
                else:
                    self.content = [add_content]
            # Accumulate (default behavior)
            elif isinstance(add_content, Sequence):
                self.content.extend(add_content)
            else:
                self.content.append(add_content)

        # Accumulate locations (never replace)
        # TODO: perhaps both should be possible?
        if add_location is not None:
            if isinstance(add_location, str):
                add_location = ToolCallLocation(path=add_location)
            self.locations.append(add_location)

        if add_locations is not None:
            for loc in add_locations:
                location = ToolCallLocation(path=loc) if isinstance(loc, str) else loc
                self.locations.append(location)

        # Send update with ALL accumulated data
        await self._notifications.tool_call_progress(
            tool_call_id=self.tool_call_id,
            status=self.status,
            title=self.title,
            kind=self.kind,
            locations=self.locations or None,
            content=self.content or None,
            raw_output=self.raw_output,
        )

    async def add_terminal(self, terminal_id: str, *, title: str | None = None) -> None:
        """Add terminal content to the tool call.

        Args:
            terminal_id: ID of the terminal to embed
            title: Optional title update
        """
        content = TerminalToolCallContent(terminal_id=terminal_id)
        await self.update(add_content=content, title=title, status="in_progress")

    async def add_text(self, text: str) -> None:
        """Add text content to the tool call.

        Args:
            text: Text to add
        """
        content = ContentToolCallContent.text(text=text)
        await self.update(add_content=content)

    async def complete(
        self,
        raw_output: Any = None,
        *,
        add_content: ToolCallContent | Sequence[ToolCallContent] | None = None,
    ) -> None:
        """Mark tool call as completed.

        Args:
            raw_output: Final output data
            add_content: Optional final content to add
        """
        await self.update(status="completed", raw_output=raw_output, add_content=add_content)

    async def fail(
        self,
        error: str | None = None,
        *,
        raw_output: Any = None,
    ) -> None:
        """Mark tool call as failed.

        Args:
            error: Error message to display
            raw_output: Optional error details
        """
        add_content = None
        if error:
            add_content = ContentToolCallContent.text(text=f"Error: {error}")
        await self.update(status="failed", add_content=add_content, raw_output=raw_output)
