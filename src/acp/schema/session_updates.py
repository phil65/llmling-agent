"""Session update schema definitions."""

from __future__ import annotations

import base64
from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import Field

from acp.schema.agent_plan import PlanEntry  # noqa: TC001
from acp.schema.base import AnnotatedObject
from acp.schema.content_blocks import (  # noqa: TC001
    Annotations,
    Audience,
    AudioContentBlock,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)
from acp.schema.slash_commands import AvailableCommand  # noqa: TC001
from acp.schema.tool_call import (  # noqa: TC001
    ToolCallContent,
    ToolCallKind,
    ToolCallLocation,
)


if TYPE_CHECKING:
    from datetime import datetime


ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]


class BaseChunk(AnnotatedObject):
    """Base class for all session update chunks."""

    content: ContentBlock
    """A single item of content"""

    @classmethod
    def text(
        cls,
        text: str,
        *,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self:
        """Create a chunk containing text.

        Args:
            text: The text content.
            audience: The audience for the text.
            last_modified: The last modified date of the text.
            priority: The priority of the text.
        """
        annotations = Annotations.optionally_create(
            audience=audience,
            last_modified=last_modified,
            priority=priority,
        )
        return cls(content=TextContentBlock(text=text, annotations=annotations))

    @classmethod
    def image(
        cls,
        data: str | bytes,
        mime_type: str,
        uri: str | None = None,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self:
        """Create a chunk containing an embedded image resource.

        Args:
            data: The image data.
            mime_type: The MIME type of the image.
            uri: The URI of the image.
            audience: The audience for the image.
            last_modified: The last modified date of the image.
            priority: The priority of the image.
        """
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode()
        annotations = Annotations.optionally_create(
            audience=audience,
            last_modified=last_modified,
            priority=priority,
        )
        image = ImageContentBlock(
            data=data,
            mime_type=mime_type,
            uri=uri,
            annotations=annotations,
        )
        return cls(content=image)

    @classmethod
    def audio(
        cls,
        data: str | bytes,
        mime_type: str,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self:
        """Create a chunk containing audio content.

        Args:
            data: The audio data.
            mime_type: The MIME type of the audio data.
            audience: The audience for the audio chunk.
            last_modified: The last modified date of the audio chunk.
            priority: The priority of the audio chunk.
        """
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode()
        annotations = Annotations.optionally_create(
            audience=audience,
            last_modified=last_modified,
            priority=priority,
        )
        content = AudioContentBlock(
            data=data,
            mime_type=mime_type,
            annotations=annotations,
        )
        return cls(content=content)

    @classmethod
    def resource(
        cls,
        name: str,
        uri: str,
        description: str | None = None,
        mime_type: str | None = None,
        size: int | None = None,
        title: str | None = None,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self:
        """Create a chunk containing a resource content block.

        Args:
            name: The name of the resource.
            uri: The URI of the resource.
            description: The description of the resource.
            mime_type: The MIME type of the resource.
            size: The size of the resource.
            title: The title of the resource.
            audience: The audience of the resource.
            last_modified: The last modified date of the resource.
            priority: The priority of the resource.
        """
        annotations = Annotations.optionally_create(
            audience=audience,
            last_modified=last_modified,
            priority=priority,
        )
        block = ResourceContentBlock(
            name=name,
            uri=uri,
            description=description,
            mime_type=mime_type,
            size=size,
            title=title,
            annotations=annotations,
        )
        return cls(content=block)

    @classmethod
    def embedded_text_resource(
        cls,
        text: str,
        uri: str,
        mime_type: str | None = None,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self:
        """Create a chunk containing an embedded text resource.

        Args:
            text: The text to embed.
            uri: The URI of the resource.
            mime_type: The MIME type of the resource.
            audience: The audience to apply to the resource.
            last_modified: The last modified date of the resource.
            priority: The priority of the resource.
        """
        annotations = Annotations.optionally_create(
            audience=audience,
            last_modified=last_modified,
            priority=priority,
        )
        contents = TextResourceContents(text=text, mime_type=mime_type, uri=uri)
        content = EmbeddedResourceContentBlock(annotations=annotations, resource=contents)
        return cls(content=content)  # pyright: ignore[reportArgumentType]

    @classmethod
    def embedded_blob_resource(
        cls,
        data: bytes | str,
        uri: str,
        mime_type: str | None = None,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self:
        """Create a chunk containing an embedded blob resource.

        Args:
            data: The data to embed.
            uri: The URI of the resource.
            mime_type: The MIME type of the resource.
            audience: The audience to apply to the resource.
            last_modified: The last modified date of the resource.
            priority: The priority of the resource.
        """
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode()
        annotations = Annotations.optionally_create(
            audience=audience,
            last_modified=last_modified,
            priority=priority,
        )
        resource = BlobResourceContents(blob=data, mime_type=mime_type, uri=uri)
        content = EmbeddedResourceContentBlock(annotations=annotations, resource=resource)
        return cls(content=content)  # pyright: ignore[reportArgumentType]


class UserMessageChunk(BaseChunk):
    """A chunk of the user's message being streamed."""

    session_update: Literal["user_message_chunk"] = Field(default="user_message_chunk", init=False)
    """User message chunk."""


class AgentMessageChunk(BaseChunk):
    """A chunk of the agent's response being streamed."""

    session_update: Literal["agent_message_chunk"] = Field(
        default="agent_message_chunk", init=False
    )
    """Agent message chunk."""


class AgentThoughtChunk(BaseChunk):
    """A chunk of the agent's internal reasoning being streamed."""

    session_update: Literal["agent_thought_chunk"] = Field(
        default="agent_thought_chunk", init=False
    )
    """Agent thought chunk."""


class ToolCallProgress(AnnotatedObject):
    """Update on the status or results of a tool call."""

    session_update: Literal["tool_call_update"] = Field(default="tool_call_update", init=False)
    """Tool call update."""

    content: Sequence[ToolCallContent] | None = None
    """Replace the content collection."""

    kind: ToolCallKind | None = None
    """Update the tool kind."""

    locations: Sequence[ToolCallLocation] | None = None
    """Replace the locations collection."""

    raw_input: Any | None = None
    """Update the raw input."""

    raw_output: Any | None = None
    """Update the raw output."""

    status: ToolCallStatus | None = None
    """Update the execution status."""

    title: str | None = None
    """Update the human-readable title."""

    tool_call_id: str
    """The ID of the tool call being updated."""


class CurrentModeUpdate(AnnotatedObject):
    """The current mode of the session has changed.

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """

    current_mode_id: str
    """The ID of the current mode"""

    session_update: Literal["current_mode_update"] = Field(
        default="current_mode_update", init=False
    )


class AgentPlanUpdate(AnnotatedObject):
    """The agent's execution plan for complex tasks.

    See protocol docs: [Agent Plan](https://agentclientprotocol.com/protocol/agent-plan).
    """

    session_update: Literal["plan"] = Field(default="plan", init=False)

    entries: Sequence[PlanEntry]
    """The list of tasks to be accomplished.

    When updating a plan, the agent must send a complete list of all entries
    with their current status. The client replaces the entire plan with each update."""


class AvailableCommandsUpdate(AnnotatedObject):
    """Available commands are ready or have changed."""

    session_update: Literal["available_commands_update"] = Field(
        default="available_commands_update", init=False
    )
    """Available commands are ready or have changed."""

    available_commands: Sequence[AvailableCommand]
    """Commands the agent can execute"""


class CurrentModelUpdate(AnnotatedObject):
    """**UNSTABLE**: This capability is not part of the spec yet.

    The current model of the session has changed.
    """

    current_model_id: str
    """The ID of the current model."""

    session_update: Literal["current_model_update"] = Field(
        default="current_model_update", init=False
    )


class ToolCallStart(AnnotatedObject):
    """Notification that a new tool call has been initiated."""

    session_update: Literal["tool_call"] = Field(default="tool_call", init=False)
    """Notification that a new tool call has been initiated."""

    content: Sequence[ToolCallContent] | None = None
    """Content produced by the tool call."""

    kind: ToolCallKind | None = None
    """The category of tool being invoked.

    Helps clients choose appropriate icons and UI treatment.
    """

    locations: Sequence[ToolCallLocation] | None = None
    """File locations affected by this tool call.

    Enables "follow-along" features in clients.
    """

    raw_input: Any | None = None
    """Raw input parameters sent to the tool."""

    raw_output: Any | None = None
    """Raw output returned by the tool."""

    status: ToolCallStatus | None = None
    """Current execution status of the tool call."""

    title: str
    """Human-readable title describing what the tool is doing."""

    tool_call_id: str
    """Unique identifier for this tool call within the session."""


SessionUpdate = Annotated[
    (
        UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AvailableCommandsUpdate
        | AgentPlanUpdate
        | CurrentModeUpdate
        | CurrentModelUpdate
    ),
    Field(discriminator="session_update"),
]
