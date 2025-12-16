"""Sophisticated file editing tool with multiple replacement strategies.

This module implements a robust file editing tool that uses multiple fallback
strategies for finding and replacing text, similar to OpenCode's edit tool.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel, Field, field_validator

from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Generator

    from llmling_agent.agents import AgentContext


class FuzzyMatch(NamedTuple):
    """Result of fuzzy text matching."""

    similarity: float
    start_line: int
    end_line: int
    text: str


class EditParams(BaseModel):
    """Parameters for the edit tool."""

    file_path: str = Field(description="The path to the file to modify")
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(description="The text to replace it with")
    replace_all: bool = Field(
        default=False, description="Replace all occurrences of old_string (default false)"
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path is not empty."""
        if not v.strip():
            msg = "file_path cannot be empty"
            raise ValueError(msg)
        return v

    def model_post_init(self, __context: Any, /) -> None:
        """Post-initialization validation."""
        if self.old_string == self.new_string:
            msg = "old_string and new_string must be different"
            raise ValueError(msg)


def _levenshtein_distance(a: str, b: str, /) -> int:
    """Calculate Levenshtein distance between two strings."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # deletion
                matrix[i][j - 1] + 1,  # insertion
                matrix[i - 1][j - 1] + cost,  # substitution
            )

    return matrix[len(a)][len(b)]


def _simple_replacer(content: str, find: str) -> Generator[str]:
    """Direct string matching replacer."""
    if find in content:
        yield find


def _line_trimmed_replacer(content: str, find: str) -> Generator[str]:
    """Line-by-line matching with trimmed whitespace."""
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    # Remove trailing empty line if present
    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    for i in range(len(original_lines) - len(search_lines) + 1):
        matches = True

        for j in range(len(search_lines)):
            if i + j >= len(original_lines):
                matches = False
                break

            original_trimmed = original_lines[i + j].strip()
            search_trimmed = search_lines[j].strip()

            if original_trimmed != search_trimmed:
                matches = False
                break

        if matches:
            # Calculate the actual substring in the original content
            start_index = sum(len(original_lines[k]) + 1 for k in range(i))
            end_index = start_index

            for k in range(len(search_lines)):
                end_index += len(original_lines[i + k])
                if k < len(search_lines) - 1:
                    end_index += 1  # newline

            yield content[start_index:end_index]


def _calculate_block_similarity(
    original_lines: list[str],
    search_lines: list[str],
    start_line: int,
    end_line: int,
    search_block_size: int,
) -> float:
    """Calculate similarity between search block and candidate block."""
    actual_block_size = end_line - start_line + 1
    lines_to_check = min(search_block_size - 2, actual_block_size - 2)

    if lines_to_check <= 0:
        return 1.0

    similarity = 0.0
    for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
        original_line = original_lines[start_line + j].strip()
        search_line = search_lines[j].strip()
        max_len = max(len(original_line), len(search_line))

        if max_len == 0:
            continue

        distance = _levenshtein_distance(original_line, search_line)
        similarity += (1 - distance / max_len) / lines_to_check

    return similarity


def _block_anchor_replacer(content: str, find: str) -> Generator[str]:
    """Multi-line block matching using first/last line anchors with similarity scoring."""
    single_candidate_threshold = 0.0
    multiple_candidates_threshold = 0.3
    min_lines_for_block = 3

    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if len(search_lines) < min_lines_for_block:
        return

    # Remove trailing empty line if present
    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    first_line_search = search_lines[0].strip()
    last_line_search = search_lines[-1].strip()
    search_block_size = len(search_lines)

    # Find all candidate positions
    candidates: list[tuple[int, int]] = []
    for i in range(len(original_lines)):
        if original_lines[i].strip() != first_line_search:
            continue

        # Look for matching last line
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_line_search:
                candidates.append((i, j))
                break

    if not candidates:
        return

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        similarity = _calculate_block_similarity(
            original_lines, search_lines, start_line, end_line, search_block_size
        )

        if similarity >= single_candidate_threshold:
            start_index = sum(len(original_lines[k]) + 1 for k in range(start_line))
            end_index = start_index + sum(
                len(original_lines[k]) + (1 if k < end_line else 0)
                for k in range(start_line, end_line + 1)
            )
            yield content[start_index:end_index]
    else:
        # Multiple candidates - find best match
        best_match = None
        max_similarity = -1.0

        for start_line, end_line in candidates:
            similarity = _calculate_block_similarity(
                original_lines, search_lines, start_line, end_line, search_block_size
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = (start_line, end_line)

        if max_similarity >= multiple_candidates_threshold and best_match:
            start_line, end_line = best_match
            start_index = sum(len(original_lines[k]) + 1 for k in range(start_line))
            end_index = start_index + sum(
                len(original_lines[k]) + (1 if k < end_line else 0)
                for k in range(start_line, end_line + 1)
            )
            yield content[start_index:end_index]


def _whitespace_normalized_replacer(content: str, find: str) -> Generator[str]:
    """Whitespace-normalized matching replacer."""

    def normalize_whitespace(text: str) -> str:
        # Normalize multiple whitespace to single space
        text = re.sub(r"\s+", " ", text).strip()
        # Normalize spacing around common punctuation
        # Normalize spacing around common punctuation
        replacements = [
            (r"\s*:\s*", ":"),
            (r"\s*;\s*", ";"),
            (r"\s*,\s*", ","),
            (r"\s*\(\s*", "("),
            (r"\s*\)\s*", ")"),
            (r"\s*\[\s*", "["),
            (r"\s*\]\s*", "]"),
            (r"\s*\{\s*", "{"),
            (r"\s*\}\s*", "}"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    normalized_find = normalize_whitespace(find)
    found_matches = set()  # Track matches to avoid duplicates

    # Try to match the entire content first
    if normalize_whitespace(content) == normalized_find and content not in found_matches:
        found_matches.add(content)
        yield content
        return

    lines = content.split("\n")
    find_lines = find.split("\n")

    # Multi-line matches
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = lines[i : i + len(find_lines)]
            block_content = "\n".join(block)
            if (
                normalize_whitespace(block_content) == normalized_find
                and block_content not in found_matches
            ):
                found_matches.add(block_content)
                yield block_content
    else:
        # Single line matches
        for line in lines:
            if normalize_whitespace(line) == normalized_find and line not in found_matches:
                found_matches.add(line)
                yield line
            else:
                # Check for substring matches
                normalized_line = normalize_whitespace(line)
                if normalized_find in normalized_line:
                    # Find actual substring using regex
                    words = find.strip().split()
                    if words:
                        pattern = r"\s+".join(re.escape(word) for word in words)
                        match = re.search(pattern, line)
                        if match and match.group(0) not in found_matches:
                            found_matches.add(match.group(0))
                            yield match.group(0)


def _indentation_flexible_replacer(content: str, find: str) -> Generator[str]:
    """Indentation-flexible matching replacer."""

    def remove_common_indentation(text: str) -> str:
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return text

        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

        return "\n".join(line[min_indent:] if line.strip() else line for line in lines)

    normalized_find = remove_common_indentation(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    for i in range(len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i : i + len(find_lines)])
        if remove_common_indentation(block) == normalized_find:
            yield block


def _escape_normalized_replacer(content: str, find: str) -> Generator[str]:
    """Escape sequence normalized matching replacer."""

    def unescape_string(text: str) -> str:
        # Handle common escape sequences
        # Handle common escape sequences
        replacements = [
            ("\\n", "\n"),
            ("\\t", "\t"),
            ("\\r", "\r"),
            ("\\'", "'"),
            ('\\"', '"'),
            ("\\`", "`"),
            ("\\\\", "\\"),
            ("\\$", "$"),
        ]
        for escaped, unescaped in replacements:
            text = text.replace(escaped, unescaped)
        return text

    unescaped_find = unescape_string(find)

    # Try direct match with unescaped find
    if unescaped_find in content:
        yield unescaped_find

    # Look for content that when unescaped matches our find string
    # This handles the case where content has escaped chars but find doesn't
    for i in range(len(content) - len(find) + 1):
        for length in range(len(find), min(len(content) - i + 1, len(find) * 3)):
            substring = content[i : i + length]
            if unescape_string(substring) == unescaped_find:
                yield substring
                break

    # Try finding escaped versions in content line by line
    lines = content.split("\n")
    find_lines = unescaped_find.split("\n")

    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if unescape_string(block) == unescaped_find:
            yield block


def _trimmed_boundary_replacer(content: str, find: str) -> Generator[str]:
    """Trimmed boundary matching replacer."""
    trimmed_find = find.strip()

    if trimmed_find == find:
        return  # Already trimmed

    # Try trimmed version
    if trimmed_find in content:
        yield trimmed_find

    # Try finding blocks where trimmed content matches
    lines = content.split("\n")
    find_lines = find.split("\n")

    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed_find:
            yield block


def _context_aware_replacer(content: str, find: str) -> Generator[str]:
    """Context-aware matching using anchor lines."""
    find_lines = find.split("\n")
    min_context_lines = 3
    if len(find_lines) < min_context_lines:
        return

    # Remove trailing empty line if present
    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    content_lines = content.split("\n")
    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()

    # Find blocks with matching first and last lines
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue

        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                block_lines = content_lines[i : j + 1]

                # Check similarity of middle content
                if len(block_lines) == len(find_lines):
                    matching_lines = 0
                    total_non_empty = 0

                    for k in range(1, len(block_lines) - 1):
                        block_line = block_lines[k].strip()
                        find_line = find_lines[k].strip()

                        if block_line or find_line:
                            total_non_empty += 1
                            if block_line == find_line:
                                matching_lines += 1

                    # Require at least 50% similarity
                    min_similarity_ratio = 0.5
                    if (
                        total_non_empty == 0
                        or matching_lines / total_non_empty >= min_similarity_ratio
                    ):
                        yield "\n".join(block_lines)
                        break
                break


def _multi_occurrence_replacer(content: str, find: str) -> Generator[str]:
    """Multi-occurrence replacer that yields all exact matches."""
    start_index = 0
    while True:
        index = content.find(find, start_index)
        if index == -1:
            break
        yield find
        start_index = index + len(find)


def _trim_diff(diff_text: str) -> str:
    """Trim common indentation from diff output."""
    lines = diff_text.split("\n")
    content_lines = [
        line
        for line in lines
        if line.startswith(("+", "-", " ")) and not line.startswith(("---", "+++"))
    ]

    if not content_lines:
        return diff_text

    # Find minimum indentation
    min_indent = float("inf")
    for line in content_lines:
        content = line[1:]  # Remove +/- prefix
        if content.strip():
            indent = len(content) - len(content.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent == float("inf") or min_indent == 0:
        return diff_text

    # Trim indentation
    trimmed_lines = []
    for line in lines:
        if line.startswith(("+", "-", " ")) and not line.startswith(("---", "+++")):
            prefix = line[0]
            content = line[1:]
            trimmed_lines.append(prefix + content[int(min_indent) :])
        else:
            trimmed_lines.append(line)

    return "\n".join(trimmed_lines)


def replace_content(
    content: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """Replace content using multiple fallback strategies with detailed error messages."""
    if old_string == new_string:
        msg = "old_string and new_string must be different"
        raise ValueError(msg)

    replacers = [
        _simple_replacer,
        _line_trimmed_replacer,
        _block_anchor_replacer,
        _whitespace_normalized_replacer,
        _indentation_flexible_replacer,
        _escape_normalized_replacer,
        _trimmed_boundary_replacer,
        _context_aware_replacer,
        _multi_occurrence_replacer,
    ]

    found_matches = False

    for replacer in replacers:
        matches = list(replacer(content, old_string))
        if not matches:
            continue

        found_matches = True

        for search_text in matches:
            index = content.find(search_text)
            if index == -1:
                continue

            if replace_all:
                return content.replace(search_text, new_string)

            # Check if there are multiple occurrences
            last_index = content.rfind(search_text)
            if index != last_index:
                continue  # Multiple occurrences, need more context

            # Single occurrence - replace it
            return content[:index] + new_string + content[index + len(search_text) :]

    if not found_matches:
        # Provide helpful error with fuzzy match context
        error_msg = _build_not_found_error(content, old_string)
        raise ValueError(error_msg)

    msg = (
        "old_string found multiple times and requires more code context "
        "to uniquely identify the intended match"
    )
    raise ValueError(msg)


def _find_best_fuzzy_match(
    content: str, search_text: str, threshold: float = 0.8
) -> FuzzyMatch | None:
    """Find the best fuzzy match for search text in content."""
    content_lines = content.split("\n")
    search_lines = search_text.split("\n")
    window_size = len(search_lines)

    if window_size == 0:
        return None

    # Find non-empty lines as anchors
    non_empty_search = [line for line in search_lines if line.strip()]
    if not non_empty_search:
        return None

    first_anchor = non_empty_search[0]

    # Find candidate starting positions
    candidate_starts = set()
    spread = 5

    for i, line in enumerate(content_lines):
        if first_anchor in line:
            start_min = max(0, i - spread)
            start_max = min(len(content_lines) - window_size + 1, i + spread + 1)
            for s in range(start_min, start_max):
                candidate_starts.add(s)

    if not candidate_starts:
        # Sample first 100 positions
        max_positions = min(len(content_lines) - window_size + 1, 100)
        candidate_starts = set(range(max_positions))

    best_match = None
    best_similarity = 0.0

    for start in candidate_starts:
        end = start + window_size
        window_text = "\n".join(content_lines[start:end])

        matcher = difflib.SequenceMatcher(None, search_text, window_text)
        similarity = matcher.ratio()

        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = FuzzyMatch(
                similarity=similarity,
                start_line=start + 1,  # 1-based
                end_line=end,
                text=window_text,
            )

    return best_match


def _create_unified_diff(text1: str, text2: str) -> str:
    """Create a unified diff between two texts."""
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)

    # Ensure lines end with newline
    lines1 = [line if line.endswith("\n") else line + "\n" for line in lines1]
    lines2 = [line if line.endswith("\n") else line + "\n" for line in lines2]

    diff = difflib.unified_diff(
        lines1, lines2, fromfile="SEARCH", tofile="CLOSEST MATCH", lineterm="", n=3
    )

    result = "".join(diff)

    # Truncate if too long
    max_chars = 2000
    if len(result) > max_chars:
        result = result[:max_chars] + "\n...(diff truncated)"

    return result.rstrip()


def _build_not_found_error(content: str, old_string: str) -> str:
    """Build a helpful error message when old_string is not found."""
    lines = content.split("\n")
    search_lines = old_string.split("\n")

    error_parts = ["Search text not found in file."]

    # Add first line context
    if search_lines and search_lines[0].strip():
        first_search_line = search_lines[0].strip()
        matches = [i for i, line in enumerate(lines) if first_search_line in line]

        if matches:
            error_parts.append(
                f"\nFirst search line '{first_search_line}' appears at line(s): "
                f"{', '.join(str(i + 1) for i in matches[:3])}"
            )
        else:
            error_parts.append(
                f"\nFirst search line '{first_search_line}' not found anywhere in file"
            )

    # Try fuzzy matching
    fuzzy_match = _find_best_fuzzy_match(content, old_string, threshold=0.8)
    if fuzzy_match:
        similarity_pct = fuzzy_match.similarity * 100
        error_parts.append(
            f"\nClosest fuzzy match (similarity {similarity_pct:.1f}%) "
            f"at lines {fuzzy_match.start_line}–{fuzzy_match.end_line}:"  # noqa: RUF001
        )
        diff = _create_unified_diff(old_string, fuzzy_match.text)
        if diff:
            error_parts.append(f"\n{diff}")

    # Add debugging tips
    error_parts.append(
        "\n\nDebugging tips:"
        "\n1. Check for exact whitespace/indentation match"
        "\n2. Verify line endings match the file (\\r\\n vs \\n)"
        "\n3. Ensure the search text hasn't been modified"
        "\n4. Try reading the file section first to get exact text"
    )

    return "".join(error_parts)


async def edit_file_tool(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    context: AgentContext | None = None,
) -> dict[str, Any]:
    """Perform exact string replacements in files using sophisticated matching strategies.

    Supports multiple fallback approaches including whitespace normalization,
    indentation flexibility, and context-aware matching.

    Args:
        file_path: Path to the file to modify
        old_string: Text to replace
        new_string: Text to replace it with
        replace_all: Whether to replace all occurrences
        context: Agent execution context

    Returns:
        Dict with operation results including diff and any errors
    """
    if old_string == new_string:
        msg = "old_string and new_string must be different"
        raise ValueError(msg)

    # Resolve file path
    path = Path(file_path)
    if not path.is_absolute():
        # Make relative to current working directory
        path = Path.cwd() / path

    # Validate file exists and is a file
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Path is a directory, not a file: {path}"
        raise ValueError(msg)

    # Read current content
    try:
        original_content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with different encoding
        original_content = path.read_text(encoding="latin-1")

    # Handle empty file case
    if old_string == "" and original_content == "":
        new_content = new_string
    else:
        new_content = replace_content(original_content, old_string, new_string, replace_all)

    # Generate diff
    diff_lines = list(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
    )
    diff_text = "".join(diff_lines)
    trimmed_diff = _trim_diff(diff_text) if diff_text else ""

    # Write new content
    try:
        path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        msg = f"Failed to write file: {e}"
        raise RuntimeError(msg) from e

    return {
        "success": True,
        "file_path": str(path),
        "diff": trimmed_diff,
        "message": f"Successfully edited {path.name}",
        "lines_changed": len([line for line in diff_lines if line.startswith(("+", "-"))]),
    }


# Create the tool instance
edit_tool = Tool.from_callable(edit_file_tool, name_override="edit_file")
