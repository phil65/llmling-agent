"""YouTube transcript converter."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from youtube_transcript_api import YouTubeTranscriptApi  # pyright: ignore
from youtube_transcript_api.formatters import (  # pyright: ignore
    JSONFormatter,
    SRTFormatter,
    TextFormatter,
    WebVTTFormatter,
)

from llmling_agent.log import get_logger
from llmling_agent_config.converters import YouTubeConverterConfig
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


logger = get_logger(__name__)

FormatterType = Literal["text", "json", "vtt", "srt"]


class YouTubeTranscriptConverter(DocumentConverter):
    """Converter for YouTube video transcripts."""

    # YouTube URL/ID patterns
    URL_PATTERNS: ClassVar[list[re.Pattern]] = [
        # Standard YouTube URLs
        re.compile(r"https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)"),
        # Short URLs
        re.compile(r"https?://youtu\.be/([a-zA-Z0-9_-]+)"),
        # Just the video ID
        re.compile(r"^[a-zA-Z0-9_-]{11}$"),
    ]

    def __init__(self, config: YouTubeConverterConfig | None = None):
        """Initialize converter.

        Args:
            config: Configuration for the converter. If None, uses defaults.
        """
        self.config = config or YouTubeConverterConfig()

        # Initialize formatter based on format
        match self.config.format:
            case "text":
                self.formatter = TextFormatter()
            case "json":
                self.formatter = JSONFormatter()
            case "vtt":
                self.formatter = WebVTTFormatter()
            case "srt":
                self.formatter = SRTFormatter()
            case _:
                msg = f"Invalid format: {self.config.format}"
                raise ValueError(msg)

    def extract_video_id(self, url: str) -> str | None:
        """Extract YouTube video ID from URL or ID string.

        Args:
            url: YouTube URL or video ID

        Returns:
            Video ID if found, None otherwise
        """
        for pattern in self.URL_PATTERNS:
            if match := pattern.match(url):
                return match.group(1) if len(match.groups()) else match.group(0)
        return None

    def supports_file(self, path: JoinablePathLike) -> bool:
        """Check if path looks like a YouTube URL/ID."""
        return bool(self.extract_video_id(str(path)))

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Only supports URLs/IDs, not raw content."""
        return False

    def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Not supported - use convert_file instead."""
        msg = "Raw content conversion not supported"
        raise NotImplementedError(msg)

    def convert_file(self, path: JoinablePathLike) -> str:
        """Convert YouTube transcript to text.

        Args:
            path: YouTube URL or video ID

        Returns:
            Formatted transcript text

        Raises:
            ValueError: If URL/ID is invalid or transcript cannot be fetched
        """
        video_id = self.extract_video_id(str(path))
        if not video_id:
            msg = f"Invalid YouTube URL/ID: {path}"
            raise ValueError(msg)

        proxies = {"https": self.config.https_proxy} if self.config.https_proxy else None

        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id=video_id,
                languages=self.config.languages,
                preserve_formatting=self.config.preserve_formatting,
                proxies=proxies,
                cookies=self.config.cookies_path,
            )
            return self.formatter.format_transcript(transcript)

        except Exception as e:
            msg = f"Failed to fetch transcript for {path}: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e


if __name__ == "__main__":
    # Example usage
    converter = YouTubeTranscriptConverter()
    print(converter.convert_file("dQw4w9WgXcQ"))
    print(converter.convert_file("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
