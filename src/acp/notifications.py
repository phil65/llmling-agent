"""ACP notification helper for clean session update API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AvailableCommand,
    AvailableCommandsUpdate,
    BlobResourceContents,
    ContentToolCallContent,
    # CurrentModelUpdate,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    ImageContentBlock,
    ResourceContentBlock,
    SessionNotification,
    TerminalToolCallContent,
    TextContentBlock,
    TextResourceContents,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from acp.utils import infer_tool_kind
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import ModelRequest, ModelResponse

    from acp import Client
    from acp.schema import (
        Annotations,
        AvailableCommand,
        PlanEntry,
        ToolCallKind,
        ToolCallStatus,
    )
    from acp.schema.tool_call import ToolCallContent

    ContentType = Sequence[ToolCallContent | str]
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
        self._tool_call_inputs: dict[str, dict] = {}

    async def tool_call(
        self,
        tool_name: str,
        *,
        tool_input: dict[str, Any],
        tool_output: Any,
        status: ToolCallStatus = "completed",
        tool_call_id: str | None = None,
    ) -> None:
        """Send tool execution as ACP tool call update.

        Args:
            tool_name: Name of the tool that was executed
            tool_input: Input parameters passed to the tool
            tool_output: Output returned by the tool
            status: Execution status
            tool_call_id: Tool call identifier

        Returns:
            SessionNotification with tool call update
        """
        # Create tool call content from output
        content: list[ContentToolCallContent] = []
        if tool_output is not None:
            # Handle pre-converted raw content blocks
            if isinstance(tool_output, list) and all(
                isinstance(item, (TextContentBlock, ImageContentBlock, AudioContentBlock))
                for item in tool_output
            ):
                # Wrap raw blocks in ContentToolCallContent
                content = [ContentToolCallContent(content=block) for block in tool_output]
            else:
                # Fallback to string conversion
                output_text = str(tool_output)
                block = TextContentBlock(text=output_text)
                content.append(ContentToolCallContent(content=block))

        # Extract file locations if present
        locations = [
            ToolCallLocation(path=value)
            for key, value in tool_input.items()
            if key in {"path", "file_path", "filepath"} and isinstance(value, str)
        ]

        # Use appropriate notification type based on status
        if status == "pending":
            await self.tool_call_start(
                tool_call_id=tool_call_id or f"{tool_name}_{hash(str(tool_input))}",
                title=f"Execute {tool_name}",
                kind=infer_tool_kind(tool_name),
                locations=locations or None,
                content=content or None,
                raw_input=tool_input,
            )
        else:
            # For in_progress, completed, and failed statuses
            await self.tool_call_progress(
                tool_call_id=tool_call_id or f"{tool_name}_{hash(str(tool_input))}",
                title=f"Execute {tool_name}",
                status=status,
                locations=locations or None,
                content=content or None,
                raw_output=tool_output,
            )

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
        raw_output: Any | None = None,
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
        plan = AgentPlanUpdate(entries=entries)
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

    async def send_agent_thought(self, message: str) -> None:
        """Send a text message notification.

        Args:
            message: Text message to send
        """
        update = AgentThoughtChunk(content=TextContentBlock(text=message))
        notification = SessionNotification(session_id=self.id, update=update)
        await self.client.session_update(notification)

    async def send_user_message(self, message: str) -> None:
        """Send a user message notification.

        Args:
            message: Text message to send
        """
        update = UserMessageChunk(content=TextContentBlock(text=message))
        notification = SessionNotification[Any](session_id=self.id, update=update)
        await self.client.handle_notification(notification)

    async def send_user_image(
        self,
        data: str,
        mime_type: str,
        *,
        uri: str | None = None,
        annotations: Any | None = None,
    ) -> None:
        """Send a user image notification.

        Args:
            data: Base64-encoded image data
            mime_type: MIME type of the image
            uri: Optional URI of the image
            annotations: Optional annotations for the content block
        """
        content = ImageContentBlock(
            data=data,
            mime_type=mime_type,
            uri=uri,
            annotations=annotations,
        )
        update = UserMessageChunk(content=content)
        notification = SessionNotification[Any](session_id=self.id, update=update)
        await self.client.handle_notification(notification)

    async def send_user_audio(
        self,
        data: str,
        mime_type: str,
        *,
        annotations: Any | None = None,
    ) -> None:
        """Send a user audio notification.

        Args:
            data: Base64-encoded audio data
            mime_type: MIME type of the audio
            annotations: Optional annotations for the content block
        """
        content = AudioContentBlock(
            data=data,
            mime_type=mime_type,
            annotations=annotations,
        )
        update = UserMessageChunk(content=content)
        notification = SessionNotification[Any](session_id=self.id, update=update)
        await self.client.handle_notification(notification)

    async def send_user_resource(
        self,
        uri: str,
        name: str,
        *,
        description: str | None = None,
        mime_type: str | None = None,
        size: int | None = None,
        title: str | None = None,
        annotations: Any | None = None,
    ) -> None:
        """Send a user resource link notification.

        Args:
            uri: URI of the resource
            name: Name of the resource
            description: Optional description of the resource
            mime_type: Optional MIME type of the resource
            size: Optional size of the resource in bytes
            title: Optional title of the resource
            annotations: Optional annotations for the content block
        """
        content = ResourceContentBlock(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            size=size,
            title=title,
            annotations=annotations,
        )
        update = UserMessageChunk(content=content)
        notification = SessionNotification[Any](session_id=self.id, update=update)
        await self.client.handle_notification(notification)

    async def replay(self, messages: Sequence[ModelRequest | ModelResponse]) -> None:
        """Replay a sequence of model messages as notifications.

        Args:
            messages: Sequence of ModelRequest and ModelResponse objects to replay
        """
        from pydantic_ai import ModelRequest, ModelResponse

        for message in messages:
            try:
                match message:
                    case ModelRequest():
                        await self._replay_request(message)
                    case ModelResponse():
                        await self._replay_response(message)
                    case _:
                        logger.debug(
                            "Unhandled message type", message_type=type(message).__name__
                        )
            except Exception as e:
                logger.exception("Failed to replay message", error=str(e))

    async def _replay_request(self, request: ModelRequest) -> None:
        """Replay a ModelRequest by converting it to appropriate ACP notifications."""
        from pydantic_ai import ToolReturnPart, UserPromptPart

        from llmling_agent_acp.converters import to_acp_content_blocks

        for part in request.parts:
            match part:
                case UserPromptPart(content=content):
                    # Handle both str and Sequence[UserContent] types
                    if isinstance(content, str):
                        await self.send_user_message(content)
                    else:
                        # Convert multi-modal content to appropriate ACP content blocks
                        from llmling_agent_acp.converters import to_acp_content_blocks

                        converted_content = to_acp_content_blocks(content)
                        # Send each content block as separate notifications
                        for block in converted_content:
                            # Handle different content block types with proper
                            # pattern matching
                            match block:
                                case TextContentBlock(text=text):
                                    await self.send_user_message(text)
                                case ImageContentBlock() as img_block:
                                    await self.send_user_image(
                                        data=img_block.data,
                                        mime_type=img_block.mime_type,
                                        uri=img_block.uri,
                                        annotations=img_block.annotations,
                                    )
                                case AudioContentBlock() as audio_block:
                                    await self.send_user_audio(
                                        data=audio_block.data,
                                        mime_type=audio_block.mime_type,
                                        annotations=audio_block.annotations,
                                    )
                                case ResourceContentBlock() as resource_block:
                                    await self.send_user_resource(
                                        uri=resource_block.uri,
                                        name=resource_block.name,
                                        description=resource_block.description,
                                        mime_type=resource_block.mime_type,
                                        size=resource_block.size,
                                        title=resource_block.title,
                                        annotations=resource_block.annotations,
                                    )
                                case EmbeddedResourceContentBlock() as embedded_block:
                                    # Handle embedded resources with proper
                                    # pattern matching
                                    match embedded_block.resource:
                                        case TextResourceContents(text=text):
                                            await self.send_user_message(text)
                                        case BlobResourceContents() as blob_resource:
                                            blob_size = len(blob_resource.blob) * 3 // 4
                                            size_mb = blob_size / (1024 * 1024)
                                            mime = blob_resource.mime_type or "unknown"
                                            msg = (
                                                f"Embedded resource: {mime} "
                                                f"({size_mb:.2f} MB)"
                                            )
                                            await self.send_user_message(msg)
                                        case _:
                                            await self.send_user_message(
                                                "Embedded resource"
                                            )
                                case _:
                                    # Fallback for unknown types
                                    await self.send_user_message(str(block))

                case ToolReturnPart(
                    content=content, tool_name=tool_name, tool_call_id=tool_call_id
                ):
                    converted_content = to_acp_content_blocks(content)
                    tool_input = self._tool_call_inputs.get(tool_call_id, {})
                    await self.tool_call(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=converted_content,
                        status="completed",
                        tool_call_id=tool_call_id,
                    )
                    # Clean up stored input
                    self._tool_call_inputs.pop(tool_call_id, None)

                case _:
                    logger.debug(
                        "Unhandled request part type", part_type=type(part).__name__
                    )

    async def _replay_response(self, response: ModelResponse) -> None:
        """Replay a ModelResponse by converting it to appropriate ACP notifications."""
        from pydantic_ai import TextPart, ThinkingPart, ToolCallPart

        for part in response.parts:
            match part:
                case TextPart(content=content):
                    await self.send_agent_text(content)

                case ThinkingPart(content=content):
                    await self.send_agent_thought(content)

                case ToolCallPart(tool_call_id=tool_call_id):
                    # Store tool call inputs for later use with ToolReturnPart
                    tool_input = part.args_as_dict()
                    self._tool_call_inputs[tool_call_id] = tool_input
                    # Skip sending notification - ACP protocol overrides previous
                    # tool call state

                case _:
                    logger.debug(
                        "Unhandled response part type", part_type=type(part).__name__
                    )

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

    # async def update_session_model(self, model_id: str) -> None:
    #     """Send a session model update notification.

    #     Args:
    #         model_id: Unique identifier for the model
    #     """
    #     update = CurrentModelUpdate(current_model_id=model_id)
    #     notification = SessionNotification(session_id=self.id, update=update)
    #     await self.client.session_update(notification)

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
