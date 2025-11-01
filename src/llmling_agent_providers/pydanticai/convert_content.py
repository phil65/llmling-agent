"""Converters for Content types to pydantic-ai formats."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
import warnings

from pydantic_ai import AudioUrl, BinaryContent, DocumentUrl, ImageUrl

from llmling_agent.models import content as own_content


if TYPE_CHECKING:
    from pydantic_ai import UserContent

    from llmling_agent.models.content import Content


def content_to_pydantic_ai(content: Content | str) -> UserContent:  # noqa: PLR0911
    """Convert our Content types to pydantic-ai content types.

    Args:
        content: One of our Content instances to convert

    Returns:
        Converted pydantic-ai content or None if conversion not possible

    Note:
        Some content types may not have an equivalent in pydantic-ai.
        In these cases, None is returned and a warning is emitted.
    """
    match content:
        case str():
            return content
        case own_content.ImageURLContent(url=url):
            return ImageUrl(url=url)

        case own_content.ImageBase64Content(data=data, mime_type=mime_type):
            binary_data = base64.b64decode(data)
            return BinaryContent(data=binary_data, media_type=mime_type or "image/jpeg")
        case own_content.PDFURLContent(url=url):
            return DocumentUrl(url=url)
        case own_content.PDFBase64Content(data=data):
            binary_data = base64.b64decode(data)
            return BinaryContent(binary_data, media_type="application/pdf")
        case own_content.AudioURLContent(url=url):
            return AudioUrl(url=url)
        case own_content.AudioBase64Content(data=data, format=format):
            binary_data = base64.b64decode(data)
            return BinaryContent(data=binary_data, media_type=f"audio/{format or 'mp3'}")
        case own_content.VideoURLContent():
            msg = "Video content not supported by pydantic-ai"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return content.url
        case _:
            msg = f"Unsupported content type: {type(content)}"
            raise TypeError(msg)
