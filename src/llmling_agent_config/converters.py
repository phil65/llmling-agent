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

    type: str = Field(
        init=False,
        title="Converter type",
    )
    """Type discriminator for converter configs."""

    enabled: bool = Field(
        default=True,
        title="Converter enabled",
    )
    """Whether this converter is currently active."""

    model_config = ConfigDict(frozen=True)

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        raise NotImplementedError


class MarkItDownConfig(BaseConverterConfig):
    """Configuration for MarkItDown-based converter."""

    type: Literal["markitdown"] = Field("markitdown", init=False)
    """Type discriminator for MarkItDown converter."""

    max_size: int | None = Field(
        default=None,
        gt=0,
        examples=[1048576, 10485760, 52428800],
        title="Maximum file size",
    )
    """Optional size limit in bytes."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.markitdown_converter import MarkItDownConverter

        return MarkItDownConverter(self)


class YouTubeConverterConfig(BaseConverterConfig):
    """Configuration for YouTube transcript converter."""

    type: Literal["youtube"] = Field("youtube", init=False)
    """Type discriminator for converter config."""

    languages: list[str] = Field(
        default_factory=lambda: ["en"],
        examples=[["en"], ["en", "es", "fr"], ["de", "en"]],
        title="Preferred languages",
    )
    """Preferred language codes in priority order. Defaults to ['en']."""

    format: FormatterType = Field(
        default="text",
        examples=["text", "json", "vtt", "srt"],
        title="Output format",
    )
    """Output format. One of: text, json, vtt, srt."""

    preserve_formatting: bool = Field(
        default=False,
        title="Preserve HTML formatting",
    )
    """Whether to keep HTML formatting elements like <i> and <b>."""

    max_retries: int = Field(
        default=3,
        ge=0,
        examples=[1, 3, 5],
        title="Maximum retries",
    )
    """Maximum number of retries for failed requests."""

    timeout: int = Field(
        default=30,
        gt=0,
        examples=[15, 30, 60],
        title="Request timeout",
    )
    """Request timeout in seconds."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.youtubeconverter import YouTubeTranscriptConverter

        return YouTubeTranscriptConverter(self)


class LocalWhisperConfig(BaseConverterConfig):
    """Configuration for local Whisper model."""

    type: Literal["local_whisper"] = Field("local_whisper", init=False)
    """Type discriminator for converter config."""

    model: str | None = Field(
        default=None,
        examples=["whisper-1", "whisper-large"],
        title="Whisper model",
    )
    """Optional model name."""

    model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base",
        examples=["tiny", "base", "small", "medium", "large"],
        title="Model size",
    )
    """Size of the Whisper model to use."""

    device: Literal["cpu", "cuda"] | None = Field(
        default=None,
        examples=["cpu", "cuda"],
        title="Compute device",
    )
    """Device to run model on (None for auto-select)."""

    compute_type: Literal["float32", "float16"] = Field(
        default="float16",
        examples=["float32", "float16"],
        title="Compute precision",
    )
    """Compute precision to use."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.local_whisper import LocalWhisperConverter

        return LocalWhisperConverter(self)


class WhisperAPIConfig(BaseConverterConfig):
    """Configuration for OpenAI's Whisper API."""

    type: Literal["whisper_api"] = Field("whisper_api", init=False)
    """Type discriminator for converter config."""

    model: str | None = Field(
        default=None,
        examples=["whisper-1", "whisper-large-v2"],
        title="OpenAI model",
    )
    """Optional model name."""

    api_key: SecretStr | None = Field(
        default=None,
        title="OpenAI API key",
    )
    """OpenAI API key."""

    language: str | None = Field(
        default=None,
        examples=["en", "es", "fr", "de"],
        title="Language code",
    )
    """Optional language code."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.whisper_api import WhisperAPIConverter

        return WhisperAPIConverter(self)


class GoogleSpeechConfig(BaseConverterConfig):
    """Configuration for Google Cloud Speech-to-Text."""

    type: Literal["google_speech"] = Field("google_speech", init=False)
    """Type discriminator for converter config."""

    language: str = Field(
        default="en-US",
        examples=["en-US", "es-ES", "fr-FR", "de-DE"],
        title="Language code",
    )
    """Language code for transcription."""

    model: str = Field(
        default="default",
        examples=["default", "command_and_search", "phone_call"],
        title="Speech model",
    )
    """Speech model to use."""

    encoding: GoogleSpeechEncoding = Field(
        default="LINEAR16",
        examples=["LINEAR16", "FLAC", "MP3"],
        title="Audio encoding",
    )
    """Audio encoding format."""

    def get_converter(self) -> DocumentConverter:
        """Get the converter instance."""
        from llmling_agent_converters.google_speech import GoogleSpeechConverter

        return GoogleSpeechConverter(self)


class PlainConverterConfig(BaseConverterConfig):
    """Configuration for plain text fallback converter."""

    type: Literal["plain"] = Field("plain", init=False)
    """Type discriminator for plain text converter."""

    force: bool = Field(
        default=False,
        title="Force conversion",
    )
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

    providers: list[ConverterConfig] | None = Field(
        default=None,
        title="Converter providers",
    )
    """List of configured converter providers."""

    default_provider: str | None = Field(
        default=None,
        examples=["markitdown", "youtube", "whisper_api"],
        title="Default provider",
    )
    """Name of default provider for conversions."""

    max_size: int | None = Field(
        default=None,
        examples=[1048576, 10485760, 52428800],
        title="Global size limit",
    )
    """Global size limit for all converters."""

    model_config = ConfigDict(frozen=True)
