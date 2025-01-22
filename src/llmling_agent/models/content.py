"""Content types for messages."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from upath import UPath

from llmling_agent.utils.async_read import read_path


if TYPE_CHECKING:
    import PIL.Image

    from llmling_agent.common_types import StrPath


DetailLevel = Literal["high", "low", "auto"]


class BaseContent(BaseModel):
    """Base class for special content types (non-text)."""

    type: str = Field(init=False)
    """Discriminator field for content types."""

    description: str | None = None
    """Human-readable description of the content."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


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
        path: StrPath,
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
        path_obj = UPath(path)

        # For http(s) URLs, pass through as URL content
        if path_obj.protocol in ("http", "https"):
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
    """Type discriminator for URL-based images."""

    url: str
    """URL to the image."""


class ImageBase64Content(BaseImageContent):
    """Image from base64 data."""

    type: Literal["image_base64"] = Field("image_base64", init=False)
    """Type discriminator for base64-encoded images."""

    data: str
    """Base64-encoded image data."""

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

    @classmethod
    async def from_path(
        cls,
        path: StrPath,
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
        path_obj = UPath(path)

        # For http(s) URLs, pass through as URL content
        if path_obj.protocol in ("http", "https"):
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
    """Type discriminator for URL-based PDFs."""

    url: str
    """URL to the PDF document."""


class PDFBase64Content(BasePDFContent):
    """PDF from base64 data."""

    type: Literal["pdf_base64"] = Field("pdf_base64", init=False)
    """Type discriminator for base64-encoded PDFs."""

    data: str
    """Base64-encoded PDF data."""

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
    description: str | None = None


class AudioURLContent(AudioContent):
    """Audio from URL."""

    type: Literal["audio_url"] = Field("audio_url", init=False)
    url: str


class AudioBase64Content(AudioContent):
    """Audio from base64 data."""

    type: Literal["audio_base64"] = Field("audio_base64", init=False)
    data: str
    format: str | None = None  # mp3, wav, etc

    @classmethod
    def from_bytes(cls, data: bytes, audio_format: str = "mp3") -> Self:
        """Create from raw bytes."""
        return cls(data=base64.b64encode(data).decode(), format=audio_format)

    @classmethod
    def from_path(cls, path: StrPath) -> Self:
        """Create from file path with auto format detection."""
        import mimetypes

        path_obj = UPath(path)
        mime_type, _ = mimetypes.guess_type(str(path_obj))
        fmt = (
            mime_type.removeprefix("audio/")
            if mime_type and mime_type.startswith("audio/")
            else "mp3"
        )

        return cls(data=base64.b64encode(path_obj.read_bytes()).decode(), format=fmt)


Content = Annotated[
    ImageURLContent
    | ImageBase64Content
    | PDFURLContent
    | PDFBase64Content
    | AudioURLContent
    | AudioBase64Content,
    Field(discriminator="type"),
]
