"""Converter configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, SecretStr
from schemez import Schema


if TYPE_CHECKING:
    from llmling_agent_converters.base import DocumentConverter


FormatterType = Literal["text", "json", "vtt", "srt"]
GoogleSpeechEncoding = Literal["LINEAR16", "FLAC", "MP3"]


class BaseConverterConfig(Schema):
    """Base configuration for document converters."""

    type: str = Field(init=False)
    """Type discriminator for converter configs."""

    enabled: bool = True
    """Whether this converter is currently active."""

    model_config = ConfigDict(frozen=True)

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        raise NotImplementedError


class MarkItDownConfig(BaseConverterConfig):
    """Configuration for MarkItDown-based converter."""

    type: Literal["markitdown"] = Field("markitdown", init=False)
    """Type discriminator for MarkItDown converter."""

    max_size: int | None = Field(default=None, gt=0)
    """Optional size limit in bytes."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.markitdown_converter import MarkItDownConverter

        return MarkItDownConverter(self)


class YouTubeConverterConfig(BaseConverterConfig):
    """Configuration for YouTube transcript converter."""

    type: Literal["youtube"] = Field("youtube", init=False)
    """Type discriminator for converter config."""

    languages: list[str] = Field(default_factory=lambda: ["en"])
    """Preferred language codes in priority order. Defaults to ['en']."""

    format: FormatterType = "text"
    """Output format. One of: text, json, vtt, srt."""

    preserve_formatting: bool = False
    """Whether to keep HTML formatting elements like <i> and <b>."""

    cookies_path: str | None = None
    """Optional path to cookies file for age-restricted videos."""

    https_proxy: str | None = None
    """Optional HTTPS proxy URL (format: https://user:pass@domain:port)."""

    max_retries: int = Field(default=3, ge=0)
    """Maximum number of retries for failed requests."""

    timeout: int = Field(default=30, gt=0)
    """Request timeout in seconds."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.youtubeconverter import YouTubeTranscriptConverter

        return YouTubeTranscriptConverter(self)


class LocalWhisperConfig(BaseConverterConfig):
    """Configuration for local Whisper model."""

    type: Literal["local_whisper"] = Field("local_whisper", init=False)
    """Type discriminator for converter config."""

    model: str | None = None
    """Optional model name."""

    model_size: Literal["tiny", "base", "small", "medium", "large"] = "base"
    """Size of the Whisper model to use."""

    device: Literal["cpu", "cuda"] | None = None
    """Device to run model on (None for auto-select)."""

    compute_type: Literal["float32", "float16"] = "float16"
    """Compute precision to use."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.local_whisper import LocalWhisperConverter

        return LocalWhisperConverter(self)


class WhisperAPIConfig(BaseConverterConfig):
    """Configuration for OpenAI's Whisper API."""

    type: Literal["whisper_api"] = Field("whisper_api", init=False)
    """Type discriminator for converter config."""

    model: str | None = None
    """Optional model name."""

    api_key: SecretStr | None = None
    """OpenAI API key."""

    language: str | None = None
    """Optional language code."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.whisper_api import WhisperAPIConverter

        return WhisperAPIConverter(self)


class GoogleSpeechConfig(BaseConverterConfig):
    """Configuration for Google Cloud Speech-to-Text."""

    type: Literal["google_speech"] = Field("google_speech", init=False)
    """Type discriminator for converter config."""

    language: str = "en-US"
    """Language code for transcription."""

    model: str = "default"
    """Speech model to use."""

    encoding: GoogleSpeechEncoding = "LINEAR16"
    """Audio encoding format."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.google_speech import GoogleSpeechConverter

        return GoogleSpeechConverter(self)


class PlainConverterConfig(BaseConverterConfig):
    """Configuration for plain text fallback converter."""

    type: Literal["plain"] = Field("plain", init=False)
    """Type discriminator for plain text converter."""

    force: bool = False
    """Whether to attempt converting any file type."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.plain_converter import PlainConverter

        return PlainConverter(self)


ConverterConfig = Annotated[
    MarkItDownConfig
    | PlainConverterConfig
    | YouTubeConverterConfig
    | WhisperAPIConfig
    | LocalWhisperConfig
    | GoogleSpeechConfig,
    Field(discriminator="type"),
]


class ConversionConfig(Schema):
    """Global conversion configuration."""

    providers: list[ConverterConfig] | None = None
    """List of configured converter providers."""

    default_provider: str | None = None
    """Name of default provider for conversions."""

    max_size: int | None = None
    """Global size limit for all converters."""

    model_config = ConfigDict(frozen=True)
