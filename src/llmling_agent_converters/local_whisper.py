"""Local Whisper audio-to-text converter."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, ClassVar

from llmling_agent_config.converters import LocalWhisperConfig
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


class LocalWhisperConverter(DocumentConverter):
    """Converter using local Whisper model."""

    SUPPORTED_MIME_TYPES: ClassVar = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "audio/flac",
    }

    def __init__(self, config: LocalWhisperConfig | None = None):
        """Initialize converter with config."""
        self.config = config or LocalWhisperConfig()
        self._model = None

    def _ensure_model(self):
        """Load model if not already loaded."""
        if self._model is not None:
            return

        import torch  # pyright: ignore
        import whisper  # pyright: ignore

        device = self.config.device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = whisper.load_model(
            self.config.model_size,
            device=device,
        )

    def supports_file(self, path: JoinablePathLike) -> bool:
        """Check if file type is supported."""
        import mimetypes

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

        self._ensure_model()
        import whisper  # pyright: ignore

        audio = whisper.load_audio(io.BytesIO(content))
        assert self._model
        is_fp16 = self.config.compute_type == "float16"
        result = self._model.transcribe(audio, fp16=(is_fp16))
        return result["text"]
