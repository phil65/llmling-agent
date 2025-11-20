"""Content block schema definitions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field

from acp.schema.base import AnnotatedObject


Audience = Sequence[Literal["assistant", "user"]]


class BaseResourceContents(AnnotatedObject):
    """Text-based resource contents."""

    mime_type: str | None = None
    """MIME type of the resource."""

    uri: str
    """URI of the resource."""


class TextResourceContents(BaseResourceContents):
    """Text-based resource contents."""

    text: str
    """Text content of the resource."""


class BlobResourceContents(BaseResourceContents):
    """Binary resource contents."""

    blob: str
    """Base64-encoded binary content of the resource."""


class Annotations(AnnotatedObject):
    """Optional annotations for the client.

    The client can use annotations to inform how objects are used or displayed.
    """

    audience: Audience | None = None
    """Audience for the annotated resource."""

    last_modified: str | None = None
    """Last modified timestamp of the annotated resource."""

    priority: float | None = None
    """Priority of the annotated resource."""

    @classmethod
    def optionally_create(
        cls,
        audience: Audience | None = None,
        last_modified: datetime | str | None = None,
        priority: float | None = None,
    ) -> Self | None:
        """Create an annotations object if any value is given, otherwise return None."""
        if audience is None and last_modified is None and priority is None:
            return None
        if isinstance(last_modified, datetime):
            last_modified = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(audience=audience, last_modified=last_modified, priority=priority)


class BaseContentBlock(AnnotatedObject):
    """Base content block."""

    annotations: Annotations | None = None
    """Annotations for the content block."""


ResourceContents = TextResourceContents | BlobResourceContents


class EmbeddedResourceContentBlock[TResourceContents: ResourceContents = ResourceContents](
    BaseContentBlock
):
    """Complete resource contents embedded directly in the message.

    Preferred for including context as it avoids extra round-trips.

    Requires the `embeddedContext` prompt capability when included in prompts.
    """

    type: Literal["resource"] = Field(default="resource", init=False)

    resource: TResourceContents
    """Resource content that can be embedded in a message."""


class TextContentBlock(BaseContentBlock):
    """Text content.

    May be plain text or formatted with Markdown.

    All agents MUST support text content blocks in prompts.
    Clients SHOULD render this text as Markdown.
    """

    type: Literal["text"] = Field(default="text", init=False)

    text: str
    """Text content of the block."""


class ImageContentBlock(BaseContentBlock):
    """Images for visual context or analysis.

    Requires the `image` prompt capability when included in prompts.
    """

    type: Literal["image"] = Field(default="image", init=False)

    data: str
    """Base64-encoded image data."""

    mime_type: str
    """MIME type of the image."""

    uri: str | None = None
    """URI of the image."""


class AudioContentBlock(BaseContentBlock):
    """Audio data for transcription or analysis.

    Requires the `audio` prompt capability when included in prompts.
    """

    type: Literal["audio"] = Field(default="audio", init=False)

    data: str
    """Base64-encoded audio data."""

    mime_type: str
    """MIME type of the audio."""


# Resource links:
# [@index.js](file:///Users/.../projects/reqwest/examples/wasm_github_fetch/index.js)
# [@wasm](file:///Users/.../projects/reqwest/src/wasm)
# [@error](file:///Users/.../projects/reqwest/src/async_impl/client.rs?symbol=Error#L2661:2661)
# [@error.rs (23:27)](file:///Users/.../projects/reqwest/src/error.rs#L23:27)


class ResourceContentBlock(BaseContentBlock):
    """References to resources that the agent can access.

    All agents MUST support resource links in prompts.
    """

    type: Literal["resource_link"] = Field(default="resource_link", init=False)

    description: str | None = None
    """Description of the resource."""

    mime_type: str | None = None
    """MIME type of the resource."""

    name: str
    """Name of the resource."""

    size: int | None = Field(default=None, ge=0)
    """Size of the resource in bytes."""

    title: str | None = None
    """Title of the resource."""

    uri: str
    """URI of the resource."""


ContentBlock = Annotated[
    (
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ),
    Field(discriminator="type"),
]
