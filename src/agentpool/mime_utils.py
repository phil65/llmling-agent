"""Centralized MIME type utilities for agentpool.

Uses pydantic-ai's enhanced MimeTypes instance for better cross-platform
coverage of modern file types (.xlsx, .docx, .webp, .mkv, etc.).
"""

from __future__ import annotations

from pydantic_ai.messages import _mime_types


# MIME type prefixes that are definitely binary (no need to probe content)
BINARY_MIME_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "application/octet-stream",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/pdf",
    "application/x-executable",
    "application/x-sharedlib",
)

# MIME type prefixes that should be treated as text
TEXT_MIME_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
)

# How many bytes to probe for binary detection
BINARY_PROBE_SIZE = 8192


def guess_type(path: str) -> str | None:
    """Guess the MIME type of a file based on its path/extension.

    Uses pydantic-ai's enhanced MimeTypes instance which has better
    coverage for modern file types across platforms.

    Args:
        path: File path or URL to guess type for

    Returns:
        MIME type string or None if unknown
    """
    mime_type, _ = _mime_types.guess_type(path)
    return mime_type


def is_binary_mime(mime_type: str | None) -> bool:
    """Check if MIME type is known to be binary (skip content probing).

    Args:
        mime_type: MIME type string or None

    Returns:
        True if the MIME type is definitely binary
    """
    if mime_type is None:
        return False
    return any(mime_type.startswith(prefix) for prefix in BINARY_MIME_PREFIXES)


def is_text_mime(mime_type: str | None) -> bool:
    """Check if a MIME type represents text content.

    Args:
        mime_type: MIME type string or None

    Returns:
        True if the MIME type is text-based (defaults to True for unknown)
    """
    if mime_type is None:
        return True  # Default to text for unknown types
    return any(mime_type.startswith(prefix) for prefix in TEXT_MIME_PREFIXES)


def is_binary_content(data: bytes) -> bool:
    """Detect binary content by probing for null bytes.

    Uses the same heuristic as git: if the first ~8KB contains a null byte,
    the content is considered binary.

    Args:
        data: Raw bytes to check

    Returns:
        True if content appears to be binary
    """
    probe = data[:BINARY_PROBE_SIZE]
    return b"\x00" in probe
