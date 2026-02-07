"""Helper utilities for working with pydantic-ai message types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from pydantic_ai import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.messages import BaseToolCallPart

from agentpool.common_types import PathReference


if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem
    from pydantic_ai import FileUrl, MultiModalContent, UserContent
    from pydantic_ai.messages import ToolCallPartDelta


def safe_args_as_dict(
    part: BaseToolCallPart | ToolCallPartDelta,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safely extract args as dict from a tool call part.

    Models can return malformed JSON for tool arguments, especially during
    streaming when args are still being assembled. This helper catches parse
    errors and returns a fallback value.

    Args:
        part: A tool call part (complete or delta) with args to extract
        default: Value to return on parse failure. If None, returns {"_raw_args": ...}
                 with the original unparsed args.

    Returns:
        The parsed arguments dict, or a fallback on parse failure.
    """
    if not isinstance(part, BaseToolCallPart):
        # ToolCallPartDelta doesn't have args_as_dict
        if default is not None:
            return default
        raw = getattr(part, "args", None)
        return {"_raw_args": raw} if raw else {}
    try:
        return part.args_as_dict()
    except ValueError:
        # Model returned malformed JSON for tool args
        if default is not None:
            return default
        # Preserve raw args for debugging/inspection
        return {"_raw_args": part.args} if part.args else {}


def url_from_mime_type(uri: str, mime_type: str | None) -> FileUrl:
    """Convert URI to appropriate pydantic-ai URL type based on MIME type."""
    if not mime_type:
        return DocumentUrl(url=uri)

    if mime_type.startswith("image/"):
        return ImageUrl(url=uri, media_type=mime_type)
    if mime_type.startswith("audio/"):
        return AudioUrl(url=uri, media_type=mime_type)
    if mime_type.startswith("video/"):
        return VideoUrl(url=uri, media_type=mime_type)
    return DocumentUrl(url=uri, media_type=mime_type)


def get_file_url_obj(url: str, mime: str) -> MultiModalContent | None:
    if mime.startswith("image/"):
        return ImageUrl(url=url, media_type=mime)
    if mime.startswith("audio/"):
        return AudioUrl(url=url, media_type=mime)
    if mime.startswith("video/"):
        return VideoUrl(url=url, media_type=mime)
    if mime == "application/pdf":
        return DocumentUrl(url=url, media_type=mime)
    return None


def to_user_content_or_path_ref(
    mime: str,
    url: str,
    filename: str | None = None,
    fs: AsyncFileSystem | None = None,
) -> UserContent | PathReference:
    """Convert an OpenCode FilePartInput to pydantic-ai content or PathReference.

    Supports:
    - file:// URLs with text/* or directory MIME -> PathReference (deferred resolution)
    - data: URIs -> BinaryContent
    - Images (image/*) -> ImageUrl or BinaryContent
    - Documents (application/pdf) -> DocumentUrl or BinaryContent
    - Audio (audio/*) -> AudioUrl or BinaryContent
    - Video (video/*) -> VideoUrl or BinaryContent

    Args:
        mime: Mime type
        url: part URL
        filename: Optional filename
        fs: Optional async filesystem for PathReference resolution

    Returns:
        Appropriate pydantic-ai content type or PathReference
    """
    from urllib.parse import unquote, urlparse

    # Handle data: URIs - convert to BinaryContent
    if url.startswith("data:"):
        return BinaryContent.from_data_uri(url)

    # Handle file:// URLs for text files and directories -> PathReference
    if url.startswith("file://"):
        parsed = urlparse(url)
        path = unquote(parsed.path)
        # Text files and directories get deferred context resolution
        title = f"@{filename}" if filename else None
        if mime.startswith("text/") or mime == "application/x-directory" or not mime:
            return PathReference(path=path, fs=fs, mime_type=mime or None, display_name=title)

        # Media files from local filesystem - use URL types
        if content := get_file_url_obj(url, mime):
            return content
        # Unknown MIME for file:// - defer to PathReference
        return PathReference(path=path, fs=fs, mime_type=mime or None, display_name=title)
    # Handle regular URLs based on mime type. Fallback: treat as document
    return content if (content := get_file_url_obj(url, mime)) else DocumentUrl(url=url)


def format_uri_as_link(uri: str) -> str:
    """Format URI as markdown-style link similar to other ACP implementations.

    Args:
        uri: URI to format (file://, zed://, etc.)

    Returns:
        Markdown-style link in format [@name](uri)
    """
    if uri.startswith("file://"):
        path = uri[7:]  # Remove "file://"
        name = path.split("/")[-1] or path
        return f"[@{name}]({uri})"
    if uri.startswith("zed://"):
        parts = uri.split("/")
        name = parts[-1] or uri
        return f"[@{name}]({uri})"
    return uri


def _uri_to_path(uri: str) -> str | None:
    """Extract filesystem path from a file:// URI.

    Args:
        uri: URI string

    Returns:
        Filesystem path string, or None if not a file:// URI
    """
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return unquote(parsed.path)


def uri_to_path_reference(
    uri: str,
    mime_type: str | None,
    fs: AsyncFileSystem | None,
) -> PathReference | None:
    """Create a PathReference from a URI if it's a file:// reference.

    Args:
        uri: URI string
        mime_type: Optional MIME type hint
        fs: Optional async filesystem

    Returns:
        PathReference if URI is a file:// reference, None otherwise
    """
    path = _uri_to_path(uri)
    if path is None:
        return None
    name = format_uri_as_link(uri)
    return PathReference(path=path, fs=fs, mime_type=mime_type, display_name=name)
