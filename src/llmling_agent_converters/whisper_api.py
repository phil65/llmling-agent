"""Whisper API converter implementation."""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any, ClassVar

from openai import omit

from llmling_agent.log import get_logger
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from upath.types import JoinablePathLike

    from llmling_agent_config.converters import WhisperAPIConfig


logger = get_logger(__name__)


class WhisperAPIConverter(DocumentConverter):
    """Converter using OpenAI's Whisper API."""

    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/webm",
        "audio/x-wav",
        "audio/ogg",
        "audio/flac",
        "audio/m4a",
        "video/mp4",
    }

    def __init__(self, config: WhisperAPIConfig):
        """Initialize converter with config."""
        self.config = config

    def supports_file(self, path: JoinablePathLike) -> bool:
        """Check if file type is supported."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))
        return bool(mime_type and mime_type in self.SUPPORTED_MIME_TYPES)

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if content type is supported."""
        return mime_type in self.SUPPORTED_MIME_TYPES

    def convert_content(self, content: bytes, mime_type: str | None = None) -> str:
        """Convert audio content to text."""
        from openai import OpenAI

        key = (
            self.config.api_key.get_secret_value()
            if self.config.api_key
            else os.getenv("OPENAI_API_KEY")
        )
        client = OpenAI(api_key=key)

        file = io.BytesIO(content)
        file.name = "audio.mp3"  # Required for MIME type detection

        response = client.audio.transcriptions.create(
            model=self.config.model or "whisper-1",
            file=file,
            language=self.config.language or omit,
        )
        return response.text
