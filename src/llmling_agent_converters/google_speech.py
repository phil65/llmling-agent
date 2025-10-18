"""Google-speech based Audio transcription converter."""

from __future__ import annotations

import mimetypes
from typing import TYPE_CHECKING, Any, ClassVar

from llmling_agent.log import get_logger
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from upath.types import JoinablePathLike

    from llmling_agent_config.converters import GoogleSpeechConfig

logger = get_logger(__name__)


class GoogleSpeechConverter(DocumentConverter):
    """Converter using Google Cloud Speech-to-Text."""

    SUPPORTED_MIME_TYPES: ClassVar = {"audio/wav", "audio/x-wav", "audio/flac"}

    def __init__(self, config: GoogleSpeechConfig):
        """Initialize converter with config."""
        self.config = config
        self._client = None

    def _ensure_client(self):
        """Initialize client if needed."""
        if self._client is not None:
            return

        from google.cloud import speech  # pyright: ignore

        self._client = speech.SpeechClient()

    def supports_file(self, path: JoinablePathLike) -> bool:
        """Check if file type is supported."""
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type in self.SUPPORTED_MIME_TYPES

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if content type is supported."""
        return mime_type in self.SUPPORTED_MIME_TYPES

    def convert_content(self, content: bytes, mime_type: str | None = None) -> str:
        """Convert audio content to text."""
        if not self.supports_content(content, mime_type):
            msg = f"Unsupported audio format: {mime_type}"
            raise ValueError(msg)

        self._ensure_client()
        from google.cloud import speech  # pyright: ignore

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=getattr(
                speech.RecognitionConfig.AudioEncoding, self.config.encoding
            ),
            language_code=self.config.language,
            model=self.config.model,
        )
        assert self._client
        response = self._client.recognize(config=config, audio=audio)
        return " ".join(result.alternatives[0].transcript for result in response.results)
