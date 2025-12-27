"""Grep functionality for filesystem search operations."""

from __future__ import annotations

from enum import StrEnum, auto
import re
from typing import TYPE_CHECKING, Any

from agentpool.log import get_logger
from agentpool_toolsets.fsspec_toolset.helpers import DEFAULT_MAX_SIZE


if TYPE_CHECKING:
    from exxec.base import ExecutionEnvironment
    from fsspec.asyn import AsyncFileSystem


logger = get_logger(__name__)


class GrepBackend(StrEnum):
    """Available grep backends."""

    RIPGREP = auto()
    GNU_GREP = auto()
    PYTHON = auto()


# Default patterns to exclude from grep searches
DEFAULT_EXCLUDE_PATTERNS = [
    ".venv/",
    "venv/",
    ".env/",
    "env/",
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".tox/",
    ".nox/",
    ".coverage/",
    "htmlcov/",
    "dist/",
    "build/",
    ".idea/",
    ".vscode/",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
]


async def detect_grep_backend(env: ExecutionEnvironment) -> GrepBackend:
    """Detect available grep backend in the execution environment.

    Args:
        env: ExecutionEnvironment to check for available tools

    Returns:
        Best available backend (ripgrep > GNU grep > Python)
    """
    from anyenv.os_commands import get_os_command_provider

    provider = get_os_command_provider(env.os_type)
    which_cmd = provider.get_command("which")

    # Try ripgrep first
    cmd = which_cmd.create_command("rg")
    result = await env.execute_command(cmd)
    if which_cmd.parse_command(result.stdout or "", result.exit_code or 0):
        return GrepBackend.RIPGREP

    # Try GNU grep
    cmd = which_cmd.create_command("grep")
    result = await env.execute_command(cmd)
    if which_cmd.parse_command(result.stdout or "", result.exit_code or 0):
        return GrepBackend.GNU_GREP

    return GrepBackend.PYTHON


async def grep_with_subprocess(
    env: ExecutionEnvironment,
    pattern: str,
    path: str,
    *,
    backend: GrepBackend | None = None,
    case_sensitive: bool = False,
    max_matches: int = 100,
    max_output_bytes: int = DEFAULT_MAX_SIZE,
    exclude_patterns: list[str] | None = None,
    use_gitignore: bool = True,
    context_lines: int = 0,
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute grep using ExecutionEnvironment (ripgrep or GNU grep).

    Args:
        env: ExecutionEnvironment to run the command in
        pattern: Regex pattern to search for
        path: Directory or file path to search
        backend: Grep backend to use (auto-detected if None)
        case_sensitive: Whether search is case-sensitive
        max_matches: Maximum number of matches per file
        max_output_bytes: Maximum total output bytes
        exclude_patterns: Patterns to exclude from search
        use_gitignore: Whether to respect .gitignore files
        context_lines: Number of context lines before/after match
        timeout: Command timeout in seconds

    Returns:
        Dictionary with matches, match_count, and was_truncated flag
    """
    if backend is None:
        backend = await detect_grep_backend(env)

    if backend == GrepBackend.PYTHON:
        msg = "Subprocess grep requested but no grep/ripgrep found"
        raise ValueError(msg)

    exclude = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS

    if backend == GrepBackend.RIPGREP:
        cmd_list = _build_ripgrep_command(
            pattern, path, case_sensitive, max_matches, exclude, use_gitignore, context_lines
        )
    else:
        cmd_list = _build_gnu_grep_command(
            pattern, path, case_sensitive, max_matches, exclude, context_lines
        )

    # Convert list to shell command string
    import shlex

    command = " ".join(shlex.quote(arg) for arg in cmd_list)

    try:
        result = await env.execute_command(command)

        # Exit code 0 = matches found, 1 = no matches, 2+ = error
        if result.exit_code not in {0, 1, None}:
            error_msg = result.stderr or f"Process exited with code {result.exit_code}"
            return {
                "error": f"grep error: {error_msg}",
                "matches": "",
                "match_count": 0,
                "was_truncated": False,
            }

        return _parse_grep_output(result.stdout or "", max_output_bytes)

    except Exception as e:
        logger.exception("Error running grep command")
        return {
            "error": f"Error running grep: {e}",
            "matches": "",
            "match_count": 0,
            "was_truncated": False,
        }


def _build_ripgrep_command(
    pattern: str,
    path: str,
    case_sensitive: bool,
    max_matches: int,
    exclude_patterns: list[str],
    use_gitignore: bool,
    context_lines: int = 0,
) -> list[str]:
    """Build ripgrep command."""
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--no-binary",
        "--max-count",
        str(max_matches + 1),  # Request one extra to detect truncation
    ]

    if not case_sensitive:
        cmd.append("--smart-case")

    if not use_gitignore:
        cmd.append("--no-ignore")

    # Add context lines if requested
    if context_lines > 0:
        cmd.extend(["--context", str(context_lines)])

    for pattern_exclude in exclude_patterns:
        cmd.extend(["--glob", f"!{pattern_exclude}"])

    cmd.extend(["-e", pattern, path])

    return cmd


def _build_gnu_grep_command(
    pattern: str,
    path: str,
    case_sensitive: bool,
    max_matches: int,
    exclude_patterns: list[str],
    context_lines: int = 0,
) -> list[str]:
    """Build GNU grep command."""
    cmd = [
        "grep",
        "-r",
        "-n",
        "-I",  # Skip binary files
        "-E",  # Extended regex
        f"--max-count={max_matches + 1}",
    ]

    if not case_sensitive and pattern.islower():
        cmd.append("-i")

    # Add context lines if requested
    if context_lines > 0:
        cmd.append(f"--context={context_lines}")

    for pattern_exclude in exclude_patterns:
        if pattern_exclude.endswith("/"):
            dir_pattern = pattern_exclude.rstrip("/")
            cmd.append(f"--exclude-dir={dir_pattern}")
        else:
            cmd.append(f"--exclude={pattern_exclude}")

    cmd.extend(["-e", pattern, path])

    return cmd


def _parse_grep_output(stdout: str, max_output_bytes: int) -> dict[str, Any]:
    """Parse grep output and apply size limits.

    Ripgrep/grep output format:
    - Match lines: `file:linenum:content` (colon after line number)
    - Context lines: `file-linenum-content` (dash after line number)
    - Group separators: `--`
    """
    if not stdout:
        return {"matches": "", "match_count": 0, "was_truncated": False}

    # Truncate to max bytes
    truncated_output = stdout[:max_output_bytes]
    was_truncated = len(stdout) > max_output_bytes

    # Count actual matches (lines with `:` after line number, not context lines with `-`)
    # Ripgrep/grep output formats:
    #   With filepath: "path/file.py:123:content" (match) vs "path/file.py-120-content" (context)
    #   Single file:   "123:content" (match) vs "120-content" (context)
    #   Windows:       "C:\path\file.py:123:content"
    # Pattern matches: starts with digits followed by colon, OR contains colon-digits-colon
    match_line_pattern = re.compile(r"^\d+:|:\d+:")
    match_count = sum(1 for line in stdout.splitlines() if match_line_pattern.search(line))

    return {
        "matches": truncated_output,
        "match_count": match_count,
        "was_truncated": was_truncated,
    }


async def grep_with_fsspec(
    fs: AsyncFileSystem,
    pattern: str,
    path: str,
    *,
    file_pattern: str = "**/*",
    case_sensitive: bool = False,
    max_matches: int = 100,
    max_output_bytes: int = DEFAULT_MAX_SIZE,
    context_lines: int = 0,
) -> dict[str, Any]:
    """Execute grep using fsspec filesystem (Python implementation).

    Args:
        fs: FSSpec filesystem instance
        pattern: Regex pattern to search for
        path: Base directory to search in
        file_pattern: Glob pattern to filter files
        case_sensitive: Whether search is case-sensitive
        max_matches: Maximum total matches across all files
        max_output_bytes: Maximum total output bytes
        context_lines: Number of context lines before/after match

    Returns:
        Dictionary with matches grouped by file
    """
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"error": f"Invalid regex pattern: {e}"}

    try:
        glob_path = f"{path.rstrip('/')}/{file_pattern}"
        file_paths = await fs._glob(glob_path)

        matches: dict[str, list[dict[str, Any]]] = {}
        total_matches = 0
        total_bytes = 0

        for file_path in file_paths:
            if total_matches >= max_matches or total_bytes >= max_output_bytes:
                break

            # Skip directories
            if await fs._isdir(file_path):
                continue

            try:
                content = await fs._cat_file(file_path)
                # Skip binary files
                if b"\x00" in content[:8192]:
                    continue

                text = content.decode("utf-8", errors="replace")
                lines = text.splitlines()

                file_matches: list[dict[str, Any]] = []
                for line_num, line in enumerate(lines, 1):
                    if total_matches >= max_matches or total_bytes >= max_output_bytes:
                        break

                    if regex.search(line):
                        match_info: dict[str, Any] = {
                            "line_number": line_num,
                            "content": line.rstrip(),
                        }

                        if context_lines > 0:
                            start = max(0, line_num - 1 - context_lines)
                            end = min(len(lines), line_num + context_lines)
                            match_info["context_before"] = lines[start : line_num - 1]
                            match_info["context_after"] = lines[line_num:end]

                        file_matches.append(match_info)
                        total_matches += 1
                        total_bytes += len(line.encode("utf-8"))

                if file_matches:
                    matches[file_path] = file_matches  # pyright: ignore[reportArgumentType]

            except Exception as e:  # noqa: BLE001
                logger.debug("Error reading file during grep", file=file_path, error=str(e))
                continue

        was_truncated = total_matches >= max_matches or total_bytes >= max_output_bytes
    except Exception as e:
        logger.exception("Error in fsspec grep")
        return {"error": f"Grep failed: {e}"}
    else:
        return {
            "matches": matches,
            "match_count": total_matches,
            "was_truncated": was_truncated,
        }
