"""ACP notification helper for clean session update API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from acp.schema import (
    AgentMessageChunk,
    AgentPlan,
    AudioContentBlock,
    AvailableCommand,
    AvailableCommandsUpdate,
    ContentToolCallContent,
    CurrentModelUpdate,
    CurrentModeUpdate,
    FileEditToolCallContent,
    ImageContentBlock,
    ResourceContentBlock,
    SessionNotification,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp import Client
    from acp.acp_types import ToolCallKind, ToolCallStatus
    from acp.schema import Annotations, AvailableCommand, PlanEntry

    ContentType = Sequence[
        ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent | str
    ]
logger = get_logger(__name__)


class ACPNotifications:
    """Clean API for creating and sending ACP session notifications.

    Provides convenient methods for common notification patterns,
    handling both creation and sending in a single call.
    """

    def __init__(self, client: Client, session_id: str) -> None:
        """Initialize notifications helper.

        Args:
            client: ACP client and session_id
            session_id: Session identifier
        """
        self.client = client
        self.id = session_id

    async def tool_call_start(
        self,
        tool_call_id: str,
        title: str,
        *,
        kind: ToolCallKind | None = None,
        locations: Sequence[ToolCallLocation] | None = None,
        content: ContentType | None = None,
        raw_input: dict[str, Any] | None = None,
    ) -> None:
        """Send a tool call start notification.

        Args:
            tool_call_id: Tool call identifier
            title: Optional title for the start notification
            kind: Optional tool call kind
            locations: Optional sequence of file/path locations
            content: Optional sequence of content blocks
            raw_input: Optional raw input data
        """
        start = ToolCallStart(
            tool_call_id=tool_call_id,
            status="pending",
            title=title,
            kind=kind,
            locations=locations,
            content=[
                ContentToolCallContent(content=TextContentBlock(text=i))
                if isinstance(i, str)
                else i
                for i in content or []
            ],
            raw_input=raw_input,
        )
        notification = SessionNotification(session_id=self.id, update=start)
        await self.client.session_update(notification)

    async def tool_call_progress(
        self,
        tool_call_id: str,
        status: ToolCallStatus,
        *,
        title: str | None = None,
        raw_output: str | None = None,
        locations: Sequence[ToolCallLocation] | None = None,
        content: ContentType | None = None,
    ) -> None:
        """Send a generic progress notification.

        Args:
            tool_call_id: Tool call identifier
            status: Progress status
            title: Optional title for the progress update
            raw_output: Optional raw output text
            locations: Optional sequence of file/path locations
            content: Optional sequence of content blocks or strings to display
        """
        progress = ToolCallProgress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            raw_output=raw_output,
            locations=locations,
            content=[
                ContentToolCallContent(content=TextContentBlock(text=i))
                if isinstance(i, str)
                else i
                for i in content or []
            ],
        )
        notification = SessionNotification(session_id=self.id, update=progress)
        await self.client.session_update(notification)

    async def file_edit_progress(
        self,
        tool_call_id: str,
        path: str,
        old_text: str,
        new_text: str,
        *,
        status: ToolCallStatus = "completed",
        title: str | None = None,
        changed_lines: Sequence[int] | None = None,
    ) -> None:
        """Send a notification with file edit content.

        Args:
            tool_call_id: Tool call identifier
            path: File path being edited
            old_text: Original file content
            new_text: New file content
            status: Progress status (default: 'completed')
            title: Optional title
            changed_lines: List of line numbers where changes occurred (1-based)
        """
        content = FileEditToolCallContent(path=path, old_text=old_text, new_text=new_text)

        # Create locations for changed lines or fallback to file location
        if changed_lines:
            locations = [ToolCallLocation(path=path, line=i) for i in changed_lines]
        else:
            locations = [ToolCallLocation(path=path)]

        await self.tool_call_progress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            locations=locations,
            content=[content],
        )

    async def terminal_progress(
        self,
        tool_call_id: str,
        terminal_id: str,
        *,
        status: ToolCallStatus = "completed",
        title: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        """Send a notification with terminal content.

        Args:
            tool_call_id: Tool call identifier
            terminal_id: Terminal identifier
            status: Progress status (default: 'completed')
            title: Optional title
            raw_output: Optional raw output text
        """
        terminal_content = TerminalToolCallContent(terminal_id=terminal_id)
        await self.tool_call_progress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            raw_output=raw_output,
            content=[terminal_content],
        )

    async def update_plan(self, entries: list[PlanEntry]) -> None:
        """Send a plan notification.

        Args:
            entries: List of plan entries to send
        """
        plan = AgentPlan(entries=entries)
        notification = SessionNotification(session_id=self.id, update=plan)
        await self.client.session_update(notification)

    async def update_commands(self, commands: list[AvailableCommand]) -> None:
        """Send a command update notification.

        Args:
            commands: List of available commands to send
        """
        update = AvailableCommandsUpdate(available_commands=commands)
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def send_agent_text(self, message: str) -> None:
        """Send a text message notification.

        Args:
            message: Text message to send
        """
        update = AgentMessageChunk(content=TextContentBlock(text=message))
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def send_agent_image(
        self,
        data: str,
        mime_type: str,
        *,
        uri: str | None = None,
        annotations: Annotations | None = None,
    ) -> None:
        """Send an image message notification.

        Args:
            data: Base64-encoded image data
            mime_type: MIME type of the image (e.g., 'image/png')
            uri: Optional URI reference for the image
            annotations: Optional annotations for the image
        """
        content = ImageContentBlock(
            data=data, mime_type=mime_type, uri=uri, annotations=annotations
        )
        update = AgentMessageChunk(content=content)
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def update_session_mode(self, mode_id: str) -> None:
        """Send a session mode update notification.

        Args:
            mode_id: Unique identifier for the session mode
        """
        update = CurrentModeUpdate(current_mode_id=mode_id)
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def update_session_model(self, model_id: str) -> None:
        """Send a session model update notification.

        Args:
            model_id: Unique identifier for the model
        """
        update = CurrentModelUpdate(current_model_id=model_id)
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def send_agent_audio(
        self,
        data: str,
        mime_type: str,
        *,
        annotations: Annotations | None = None,
    ) -> None:
        """Send an audio message notification.

        Args:
            data: Base64-encoded audio data
            mime_type: MIME type of the audio (e.g., 'audio/wav')
            annotations: Optional annotations for the audio
        """
        content = AudioContentBlock(
            data=data, mime_type=mime_type, annotations=annotations
        )
        update = AgentMessageChunk(content=content)
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def send_agent_resource(
        self,
        name: str,
        uri: str,
        *,
        title: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        size: int | None = None,
        annotations: Annotations | None = None,
    ) -> None:
        """Send a resource reference message notification.

        Args:
            name: Name of the resource
            uri: URI of the resource
            title: Optional title for the resource
            description: Optional description of the resource
            mime_type: Optional MIME type of the resource
            size: Optional size of the resource in bytes
            annotations: Optional annotations for the resource
        """
        content = ResourceContentBlock(
            name=name,
            uri=uri,
            title=title,
            description=description,
            mime_type=mime_type,
            size=size,
            annotations=annotations,
        )
        update = AgentMessageChunk(content=content)
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)
