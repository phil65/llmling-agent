"""FSSpec filesystem toolset helpers."""

from __future__ import annotations

import difflib
import re

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
