"""FSSpec filesystem toolset helpers."""

from __future__ import annotations

import difflib
import re
from typing import Any

from pydantic_ai import ModelRetry

from llmling_agent.log import get_logger


logger = get_logger(__name__)

# MIME types that are definitely binary (don't probe, just treat as binary)
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

# How many bytes to probe for binary detection
BINARY_PROBE_SIZE = 8192

# Default maximum size for file operations (64KB)
DEFAULT_MAX_SIZE = 64_000


async def apply_structured_edits(original_content: str, edits_response: str) -> str:
    """Apply structured edits from the agent response."""
    # Parse the edits from the response
    edits_match = re.search(r"<edits>(.*?)</edits>", edits_response, re.DOTALL)
    if not edits_match:
        logger.warning("No edits block found in response")
        return original_content

    edits_content = edits_match.group(1)

    # Find all old_text/new_text pairs
    old_text_pattern = r"<old_text[^>]*>(.*?)</old_text>"
    new_text_pattern = r"<new_text>(.*?)</new_text>"

    old_texts = re.findall(old_text_pattern, edits_content, re.DOTALL)
    new_texts = re.findall(new_text_pattern, edits_content, re.DOTALL)

    if len(old_texts) != len(new_texts):
        logger.warning("Mismatch between old_text and new_text blocks")
        return original_content

    # Apply edits sequentially
    content = original_content
    applied_edits = 0

    failed_matches = []
    multiple_matches = []

    for old_text, new_text in zip(old_texts, new_texts, strict=False):
        old_cleaned = old_text.strip()
        new_cleaned = new_text.strip()

        # Check for multiple matches (ambiguity)
        match_count = content.count(old_cleaned)
        if match_count > 1:
            multiple_matches.append(old_cleaned[:50])
        elif match_count == 1:
            content = content.replace(old_cleaned, new_cleaned, 1)
            applied_edits += 1
        else:
            failed_matches.append(old_cleaned[:50])

    # Raise ModelRetry for specific failure cases
    if applied_edits == 0 and len(old_cleaned) > 0:
        msg = (
            "Some edits were produced but none of them could be applied. "
            "Read the relevant sections of the file again so that "
            "I can perform the requested edits."
        )
        raise ModelRetry(msg)

    if multiple_matches:
        matches_str = ", ".join(multiple_matches)
        msg = (
            f"<old_text> matches multiple positions in the file: {matches_str}... "
            "Read the relevant sections of the file again and extend <old_text> "
            "to be more specific."
        )
        raise ModelRetry(msg)

    logger.info("Applied structured edits", num=applied_edits, total=len(old_texts))
    return content


def get_changed_lines(original_content: str, new_content: str, path: str) -> list[str]:
    old = original_content.splitlines(keepends=True)
    new = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old, new, fromfile=path, tofile=path, lineterm=""))
    return [line for line in diff if line.startswith(("+", "-"))]


def get_changed_line_numbers(original_content: str, new_content: str) -> list[int]:
    """Extract line numbers where changes occurred for ACP UI highlighting.

    Similar to Claude Code's line tracking for precise change location reporting.
    Returns line numbers in the new content where changes happened.

    Args:
        original_content: Original file content
        new_content: Modified file content

    Returns:
        List of line numbers (1-based) where changes occurred in new content
    """
    old_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    # Use SequenceMatcher to find changed blocks
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed_line_numbers = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert", "delete"):
            # For replacements and insertions, mark lines in new content
            # For deletions, mark the position where deletion occurred
            if tag == "delete":
                # Mark the line where deletion occurred (or next line if at end)
                line_num = min(j1 + 1, len(new_lines))
                if line_num > 0:
                    changed_line_numbers.add(line_num)
            else:
                # Mark all affected lines in new content
                for line_num in range(j1 + 1, j2 + 1):  # Convert to 1-based
                    changed_line_numbers.add(line_num)

    return sorted(changed_line_numbers)


def is_definitely_binary_mime(mime_type: str | None) -> bool:
    """Check if MIME type is known to be binary (skip content probing)."""
    if mime_type is None:
        return False
    return any(mime_type.startswith(prefix) for prefix in BINARY_MIME_PREFIXES)


def is_binary_content(data: bytes) -> bool:
    """Detect binary content by probing for null bytes.

    Uses the same heuristic as git: if the first ~8KB contains a null byte,
    the content is considered binary.
    """
    probe = data[:BINARY_PROBE_SIZE]
    return b"\x00" in probe


def truncate_content(content: str, max_size: int = DEFAULT_MAX_SIZE) -> tuple[str, bool]:
    """Truncate text content to a maximum size in bytes.

    Args:
        content: Text content to truncate
        max_size: Maximum size in bytes (default: 64KB)

    Returns:
        Tuple of (truncated_content, was_truncated)
    """
    content_bytes = content.encode("utf-8")
    if len(content_bytes) <= max_size:
        return content, False

    # Truncate at byte boundary and decode safely
    truncated_bytes = content_bytes[:max_size]
    # Avoid breaking UTF-8 sequences by decoding with error handling
    truncated = truncated_bytes.decode("utf-8", errors="ignore")
    return truncated, True


def truncate_lines(
    lines: list[str], offset: int = 0, limit: int | None = None, max_bytes: int = DEFAULT_MAX_SIZE
) -> tuple[list[str], bool]:
    """Truncate lines with offset/limit and byte size constraints.

    Args:
        lines: List of text lines
        offset: Starting line index (0-based)
        limit: Maximum number of lines to include (None = no limit)
        max_bytes: Maximum total bytes (default: 64KB)

    Returns:
        Tuple of (truncated_lines, was_truncated)
    """
    # Apply offset
    start_idx = max(0, offset)
    if start_idx >= len(lines):
        return [], False

    # Apply line limit
    end_idx = min(len(lines), start_idx + limit) if limit is not None else len(lines)

    selected_lines = lines[start_idx:end_idx]

    # Apply byte limit
    result_lines: list[str] = []
    total_bytes = 0

    for line in selected_lines:
        line_bytes = len(line.encode("utf-8"))
        if total_bytes + line_bytes > max_bytes:
            # Would exceed limit - this is actual truncation
            return result_lines, True

        result_lines.append(line)
        total_bytes += line_bytes

    # Successfully returned all requested content - not truncated
    # (byte truncation already handled above with early return)
    return result_lines, False


def _format_size(size: int) -> str:
    """Format byte size as human-readable string."""
    if size < 1024:  # noqa: PLR2004
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def format_directory_listing(
    path: str,
    directories: list[dict[str, Any]],
    files: list[dict[str, Any]],
    pattern: str = "*",
) -> str:
    """Format directory listing as markdown table.

    Args:
        path: Base directory path
        directories: List of directory info dicts
        files: List of file info dicts
        pattern: Glob pattern used

    Returns:
        Formatted markdown string
    """
    lines = [f"## {path}"]
    if pattern != "*":
        lines.append(f"Pattern: `{pattern}`")
    lines.append("")

    if not directories and not files:
        lines.append("*Empty directory*")
        return "\n".join(lines)

    lines.append("| Name | Type | Size |")
    lines.append("|------|------|------|")

    # Directories first (sorted)
    for d in sorted(directories, key=lambda x: x["name"]):
        lines.append(f"| {d['name']}/ | dir | - |")  # noqa: PERF401

    # Then files (sorted)
    for f in sorted(files, key=lambda x: x["name"]):
        size_str = _format_size(f.get("size", 0))
        lines.append(f"| {f['name']} | file | {size_str} |")

    lines.append("")
    lines.append(f"*{len(directories)} directories, {len(files)} files*")

    return "\n".join(lines)
