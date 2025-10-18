"""Content types for messages."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field
from schemez import Schema
from upathtools import read_path


if TYPE_CHECKING:
    import PIL.Image
    from upath.types import JoinablePathLike


DetailLevel = Literal["high", "low", "auto"]


class BaseContent(Schema):
    """Base class for special content types (non-text)."""

    type: str = Field(init=False)
    """Discriminator field for content types."""

    description: str | None = None
    """Human-readable description of the content."""

    model_config = ConfigDict(frozen=True)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        raise NotImplementedError


class BaseImageContent(BaseContent):
    """Base for image content."""

    detail: DetailLevel | None = None
    """Detail level for image processing by vision models.
    - high: Maximum resolution (up to 2048x2048)
    - low: Lower resolution (512x512)
    - auto: Let model decide based on content
    """

    @classmethod
    async def from_path(
        cls,
        path: JoinablePathLike,
        *,
        detail: DetailLevel | None = None,
        description: str | None = None,
    ) -> ImageURLContent | ImageBase64Content:
        """Create image content from any path.

        Automatically chooses between URL and base64 based on path type.
        Downloads and converts remote content if needed.

        Args:
            path: Local path or URL to image
            detail: Optional detail level for processing
            description: Optional description of the image
        """
        from upathtools import to_upath

        path_obj = to_upath(path)

        # For http(s) URLs, pass through as URL content
        if path_obj.protocol in {"http", "https"}:
            return ImageURLContent(
                url=str(path_obj), detail=detail, description=description
            )

        # For all other paths, read and convert to base64
        data = await read_path(path_obj, mode="rb")
        content = base64.b64encode(data).decode()
        return ImageBase64Content(data=content, detail=detail, description=description)


class ImageURLContent(BaseImageContent):
    """Image from URL."""

    type: Literal["image_url"] = Field("image_url", init=False)
    """URL-based image."""

    url: str
    """URL to the image."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for vision models."""
        content = {"url": self.url, "detail": self.detail or "auto"}
        return {"type": "image_url", "image_url": content}


class ImageBase64Content(BaseImageContent):
    """Image from base64 data."""

    type: Literal["image_base64"] = Field("image_base64", init=False)
    """Base64-encoded image."""

    data: str
    """Base64-encoded image data."""

    mime_type: str = "image/jpeg"
    """MIME type of the image."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for vision models."""
        data_url = f"data:{self.mime_type};base64,{self.data}"
        content = {"url": data_url, "detail": self.detail or "auto"}
        return {"type": "image_url", "image_url": content}

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        detail: DetailLevel | None = None,
        description: str | None = None,
    ) -> ImageBase64Content:
        """Create image content from raw bytes.

        Args:
            data: Raw image bytes
            detail: Optional detail level for processing
            description: Optional description of the image
        """
        content = base64.b64encode(data).decode()
        return cls(data=content, detail=detail, description=description)

    @classmethod
    def from_pil_image(cls, image: PIL.Image.Image) -> ImageBase64Content:
        """Create content from PIL Image."""
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return cls(data=base64.b64encode(buffer.getvalue()).decode())


class BasePDFContent(BaseContent):
    """Base for PDF document content."""

    detail: DetailLevel | None = None
    """Detail level for document processing by models."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for PDF handling."""
        raise NotImplementedError

    @classmethod
    async def from_path(
        cls,
        path: JoinablePathLike,
        *,
        detail: DetailLevel | None = None,
        description: str | None = None,
    ) -> PDFURLContent | PDFBase64Content:
        """Create PDF content from any path.

        Args:
            path: Local path or URL to PDF
            detail: Optional detail level for processing
            description: Optional description of the document
        """
        from upathtools import to_upath

        path_obj = to_upath(path)

        # For http(s) URLs, pass through as URL content
        if path_obj.protocol in {"http", "https"}:
            return PDFURLContent(
                url=str(path_obj), detail=detail, description=description
            )

        # For all other paths, read and convert to base64
        data = await read_path(path_obj, mode="rb")
        content = base64.b64encode(data).decode()
        return PDFBase64Content(data=content, detail=detail, description=description)


class PDFURLContent(BasePDFContent):
    """PDF from URL."""

    type: Literal["pdf_url"] = Field("pdf_url", init=False)
    """URL-based PDF."""

    url: str
    """URL to the PDF document."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for PDF handling."""
        content = {"url": self.url, "detail": self.detail or "auto"}
        return {"type": "file", "file": content}


class PDFBase64Content(BasePDFContent):
    """PDF from base64 data."""

    type: Literal["pdf_base64"] = Field("pdf_base64", init=False)
    """Base64-data based PDF."""

    data: str
    """Base64-encoded PDF data."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for PDF handling."""
        data_url = f"data:application/pdf;base64,{self.data}"
        content = {"url": data_url, "detail": self.detail or "auto"}
        return {"type": "file", "file": content}

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        detail: DetailLevel | None = None,
        description: str | None = None,
    ) -> Self:
        """Create PDF content from raw bytes."""
        content = base64.b64encode(data).decode()
        return cls(data=content, detail=detail, description=description)


class AudioContent(BaseContent):
    """Base for audio content."""

    format: str | None = None  # mp3, wav, etc
    """Audio format."""

    description: str | None = None
    """Human-readable description of the content."""


class AudioURLContent(AudioContent):
    """Audio from URL."""

    type: Literal["audio_url"] = Field("audio_url", init=False)
    """URL-based audio."""

    url: str
    """URL to the audio."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for audio models."""
        content = {"url": self.url, "format": self.format or "auto"}
        return {"type": "audio", "audio": content}


class AudioBase64Content(AudioContent):
    """Audio from base64 data."""

    type: Literal["audio_base64"] = Field("audio_base64", init=False)
    """Base64-encoded audio."""

    data: str
    """Audio data in base64 format."""

    format: str | None = None  # mp3, wav, etc
    """Audio format."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for audio models."""
        data_url = f"data:audio/{self.format or 'mp3'};base64,{self.data}"
        content = {"url": data_url, "format": self.format or "auto"}
        return {"type": "audio", "audio": content}

    @classmethod
    def from_bytes(cls, data: bytes, audio_format: str = "mp3") -> Self:
        """Create from raw bytes."""
        return cls(data=base64.b64encode(data).decode(), format=audio_format)

    @classmethod
    def from_path(cls, path: JoinablePathLike) -> Self:
        """Create from file path with auto format detection."""
        import mimetypes

        from upathtools import to_upath

        path_obj = to_upath(path)
        mime_type, _ = mimetypes.guess_type(str(path_obj))
        fmt = (
            mime_type.removeprefix("audio/")
            if mime_type and mime_type.startswith("audio/")
            else "mp3"
        )

        return cls(data=base64.b64encode(path_obj.read_bytes()).decode(), format=fmt)

    @property
    def mime_type(self) -> str:
        """Return the MIME type of the audio."""
        return f"audio/{self.format or 'mp3'}"


class VideoContent(BaseContent):
    """Base for video content."""

    format: str | None = None
    """Video format."""

    description: str | None = None
    """Human-readable description of the content."""


class VideoURLContent(VideoContent):
    """Video from URL."""

    type: Literal["video_url"] = Field("video_url", init=False)
    """URL-based video."""

    url: str
    """URL to the video."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format for video models."""
        content = {"url": self.url, "format": self.format or "auto"}
        return {"type": "video", "video": content}


Content = Annotated[
    ImageURLContent
    | ImageBase64Content
    | PDFURLContent
    | PDFBase64Content
    | AudioURLContent
    | AudioBase64Content
    | VideoURLContent,
    Field(discriminator="type"),
]
