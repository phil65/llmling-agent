"""Conversions between internal and MCP types."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pydantic_ai import BinaryContent, FileUrl, SystemPromptPart, TextPart, UserPromptPart


if TYPE_CHECKING:
    from mcp.types import PromptMessage
    from pydantic_ai import ModelRequestPart, ModelResponsePart


def to_mcp_messages(
    part: ModelRequestPart | ModelResponsePart,
) -> list[PromptMessage]:
    """Convert internal PromptMessage to MCP PromptMessage."""
    from mcp.types import AudioContent, ImageContent, PromptMessage, TextContent

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
            messages.append(
                PromptMessage(role="user", content=TextContent(type="text", text=msg))
            )
        case TextPart(content=msg):
            messages.append(
                PromptMessage(
                    role="assistant", content=TextContent(type="text", text=msg)
                )
            )
    return messages
