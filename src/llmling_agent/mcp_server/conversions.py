"""Conversions between internal and MCP types."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, assert_never

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    FileUrl,
    ImageUrl,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
    VideoUrl,
)

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastmcp import Client
    from mcp.types import ContentBlock
    from pydantic_ai import ModelRequestPart, ModelResponsePart


logger = get_logger(__name__)


def to_mcp_messages(
    part: ModelRequestPart | ModelResponsePart,
) -> list[PromptMessage]:
    """Convert internal PromptMessage to MCP PromptMessage."""
    messages = []
    match part:
        case UserPromptPart(content=str() as c):
            content = TextContent(type="text", text=c)
            messages.append(PromptMessage(role="user", content=content))
        case UserPromptPart(content=content_items):
            for item in content_items:
                match item:
                    case BinaryContent():
                        if item.is_audio:
                            encoded = base64.b64encode(item.data).decode("utf-8")
                            audio = AudioContent(
                                type="audio", data=encoded, mimeType=item.media_type
                            )
                            messages.append(PromptMessage(role="user", content=audio))
                        elif item.is_image:
                            encoded = base64.b64encode(item.data).decode("utf-8")
                            image = ImageContent(
                                type="image", data=encoded, mimeType=item.media_type
                            )
                        messages.append(PromptMessage(role="user", content=image))
                    case FileUrl(url=url):
                        content = TextContent(type="text", text=url)
                        messages.append(PromptMessage(role="user", content=content))

        case SystemPromptPart(content=msg):
            messages.append(PromptMessage(role="user", content=TextContent(type="text", text=msg)))
        case TextPart(content=msg):
            messages.append(
                PromptMessage(role="assistant", content=TextContent(type="text", text=msg))
            )
    return messages


def _url_from_mime_type(uri: str, mime_type: str | None) -> FileUrl:
    """Convert URI to appropriate pydantic-ai URL type based on MIME type."""
    if not mime_type:
        return DocumentUrl(url=uri)

    if mime_type.startswith("image/"):
        return ImageUrl(url=uri)
    if mime_type.startswith("audio/"):
        return AudioUrl(url=uri)
    if mime_type.startswith("video/"):
        return VideoUrl(url=uri)
    return DocumentUrl(url=uri)


async def from_mcp_content(
    mcp_content: Sequence[ContentBlock | TextResourceContents | BlobResourceContents],
    client: Client[Any] | None = None,
) -> list[str | BinaryContent]:
    """Convert MCP content blocks to PydanticAI content types.

    If a FastMCP client is given, this function will try to resolve the ResourceLinks.

    """
    contents: list[Any] = []

    for block in mcp_content:
        match block:
            case TextContent(text=text):
                contents.append(text)
            case TextResourceContents(text=text):
                contents.append(text)
            case ImageContent(data=data, mimeType=mime_type):
                decoded_data = base64.b64decode(data)
                img = BinaryImage(data=decoded_data, media_type=mime_type)
                contents.append(img)
            case AudioContent(data=data, mimeType=mime_type):
                decoded_data = base64.b64decode(data)
                content = BinaryContent(data=decoded_data, media_type=mime_type)
                contents.append(content)
            case BlobResourceContents(blob=blob):
                decoded_data = base64.b64decode(blob)
                mime = "application/octet-stream"
                content = BinaryContent(data=decoded_data, media_type=mime)
                contents.append(content)
            case ResourceLink(uri=uri, mimeType=mime_type):
                if client:
                    try:
                        res = await client.read_resource(uri)
                        nested = await from_mcp_content(res, client)
                        contents.extend(nested)
                        continue
                    except Exception:  # noqa: BLE001
                        # Fallback to URL if reading fails
                        logger.warning("Failed to read resource", uri=uri)
                # Convert to appropriate URL type based on MIME type
                contents.append(_url_from_mime_type(str(uri), mime_type))
            # mypy doesnt understand exhaustivness check for "nested typing", so we nest match-case
            case EmbeddedResource(resource=resource):
                match resource:
                    case TextResourceContents(text=text):
                        contents.append(text)
                    case BlobResourceContents() as blob_resource:
                        contents.append(f"[Binary data: {blob_resource.mimeType}]")
                    case _ as unreachable:
                        assert_never(unreachable)  # ty: ignore
            case _ as unreachable:
                assert_never(unreachable)
    return contents


def content_block_as_text(content: ContentBlock) -> str:
    match content:
        case TextContent(text=text):
            return text
        case EmbeddedResource(resource=resource):
            match resource:
                case TextResourceContents() as text_contents:
                    return text_contents.text
                case BlobResourceContents() as blob_contents:
                    return f"[Resource: {blob_contents.uri}]"
                case _ as unreachable:
                    assert_never(unreachable)  # ty: ignore
        case ResourceLink(uri=uri, description=desc):
            return f"[Resource Link: {uri}] - {desc}" if desc else f"[Resource Link: {uri}]"
        case ImageContent(mimeType=mime_type):
            return f"[Image: {mime_type}]"
        case AudioContent(mimeType=mime_type):
            return f"[Audio: {mime_type}]"
        case _ as unreachable:
            assert_never(unreachable)
