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
        2. ToolCallProgressEvent → state.update(title="Running: ...", content=terminal)
        3. ToolCallProgressEvent → state.update(content=diff, locations=path)
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
        content: ToolCallContent | Sequence[ToolCallContent] | None = None,
        locations: Sequence[ToolCallLocation | str] | None = None,
        replace: bool = False,
        raw_output: Any = None,
    ) -> None:
        """Update state and send notification with ALL accumulated data.

        Args:
            title: Override the human-readable title
            status: Update execution status
            kind: Update tool kind
            content: Content items (terminals, diffs, text) to add or replace
            locations: File locations to add or replace
            replace: If True, replace all content and locations; if False, append
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
        if content is not None:
            if replace:
                if isinstance(content, Sequence):
                    self.content = list(content)
                else:
                    self.content = [content]
            elif isinstance(content, Sequence):
                self.content.extend(content)
            else:
                self.content.append(content)

        # Handle locations: replace or accumulate (tied to replace)
        if locations is not None:
            new_locations: list[ToolCallLocation] = [
                ToolCallLocation(path=loc) if isinstance(loc, str) else loc for loc in locations
            ]
            if replace:
                self.locations = new_locations
            else:
                self.locations.extend(new_locations)

        # Send update with ALL accumulated data
        # Use tool_call_start (session_update: "tool_call") when locations are present
        # to enable clickable file paths in clients (matches Claude Code behavior)
        if self.locations:
            await self._notifications.tool_call_start(
                tool_call_id=self.tool_call_id,
                title=self.title,
                kind=self.kind,
                locations=self.locations,
                content=self.content or None,
                raw_input=self.raw_input,
            )
        else:
            await self._notifications.tool_call_progress(
                tool_call_id=self.tool_call_id,
                status=self.status,
                title=self.title,
                kind=self.kind,
                locations=None,
                content=self.content or None,
                raw_output=self.raw_output,
            )

    async def add_terminal(self, terminal_id: str, *, title: str | None = None) -> None:
        """Add terminal content to the tool call.

        Args:
            terminal_id: ID of the terminal to embed
            title: Optional title update
        """
        terminal_content = TerminalToolCallContent(terminal_id=terminal_id)
        await self.update(content=terminal_content, title=title, status="in_progress")

    async def add_text(self, text: str) -> None:
        """Add text content to the tool call.

        Args:
            text: Text to add
        """
        text_content = ContentToolCallContent.text(text=text)
        await self.update(content=text_content)

    async def complete(
        self,
        raw_output: Any = None,
        *,
        content: ToolCallContent | Sequence[ToolCallContent] | None = None,
    ) -> None:
        """Mark tool call as completed.

        Args:
            raw_output: Final output data
            content: Optional final content to add
        """
        await self.update(status="completed", raw_output=raw_output, content=content)

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
        error_content = None
        if error:
            error_content = ContentToolCallContent.text(text=f"Error: {error}")
        await self.update(status="failed", content=error_content, raw_output=raw_output)
