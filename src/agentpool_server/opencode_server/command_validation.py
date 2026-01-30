"""Shell command validation and security checks.

Provides validation for shell commands to prevent dangerous operations
like destructive commands, privilege escalation, and path traversal.
"""

from __future__ import annotations

from pathlib import Path
import re
import shlex

from fastapi import HTTPException


# Patterns for dangerous commands that should always be blocked
DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Destructive file operations
    (re.compile(r"\brm\s+(-[a-zA-Z]*)?.*\s+/\s*$"), "rm on root directory"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f.*\s+/"), "recursive force delete"),
    (re.compile(r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r.*\s+/"), "recursive force delete"),
    (re.compile(r"\bmkfs\b"), "filesystem format"),
    (re.compile(r"\bdd\s+.*of=/dev/"), "direct disk write"),
    (re.compile(r">\s*/dev/sd[a-z]"), "overwrite disk device"),
    # Privilege escalation
    (re.compile(r"\bsudo\b"), "sudo command"),
    (re.compile(r"\bsu\s+-?\s*$"), "switch user"),
    (re.compile(r"\bsu\s+root\b"), "switch to root"),
    (re.compile(r"\bdoas\b"), "doas command"),
    # Remote code execution patterns
    (re.compile(r"\bcurl\b.*\|\s*(ba)?sh"), "curl pipe to shell"),
    (re.compile(r"\bwget\b.*\|\s*(ba)?sh"), "wget pipe to shell"),
    (re.compile(r"\bcurl\b.*\|\s*python"), "curl pipe to python"),
    (re.compile(r"\bwget\b.*\|\s*python"), "wget pipe to python"),
    # Fork bombs and resource exhaustion
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"\bfork\s*bomb\b", re.IGNORECASE), "fork bomb"),
    # History/credential theft
    (re.compile(r">\s*~/.bash_history"), "history manipulation"),
    (re.compile(r"cat.*\.ssh/"), "SSH key access"),
    (re.compile(r"cat.*/etc/shadow"), "shadow file access"),
    # Shutdown/reboot
    (re.compile(r"\bshutdown\b"), "shutdown command"),
    (re.compile(r"\breboot\b"), "reboot command"),
    (re.compile(r"\binit\s+0\b"), "init shutdown"),
    (re.compile(r"\binit\s+6\b"), "init reboot"),
]

# Sensitive paths that should not be accessed
SENSITIVE_PATHS = {
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssh",
    "/root",
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.config/gcloud",
}

# Patterns that indicate path traversal attempts
PATH_TRAVERSAL_PATTERN = re.compile(r"\.\.(/|\\)")


def validate_command(command: str, working_dir: str) -> None:
    """Validate a shell command for security issues.

    Args:
        command: The shell command to validate.
        working_dir: The working directory for the command.

    Raises:
        HTTPException: If the command is dangerous or restricted.
    """
    # Check for dangerous patterns
    for pattern, description in DANGEROUS_PATTERNS:
        if pattern.search(command):
            raise HTTPException(status_code=403, detail=f"Command restricted: {description}")

    # Check for sensitive path access
    command_lower = command.lower()
    for sensitive_path in SENSITIVE_PATHS:
        # Normalize ~ to actual pattern
        path_pattern = sensitive_path.replace("~", "(/home/[^/]+|~)")
        if re.search(path_pattern, command_lower):
            detail = f"Command restricted: access to sensitive path {sensitive_path}"
            raise HTTPException(status_code=403, detail=detail)

    # Check for path traversal in the command
    if PATH_TRAVERSAL_PATTERN.search(command):
        # Parse the command to check if traversal escapes working_dir
        _check_path_traversal(command, working_dir)


def _check_path_traversal(command: str, working_dir: str) -> None:
    """Check if command contains path traversal that escapes working directory.

    Args:
        command: The shell command to check.
        working_dir: The working directory.

    Raises:
        HTTPException: If path traversal escapes the working directory.
    """
    working_path = Path(working_dir).resolve()
    # Try to extract paths from the command
    try:
        tokens = shlex.split(command)
    except ValueError:
        # If we can't parse, be conservative
        tokens = command.split()

    for token in tokens:
        # Skip flags and operators
        if token.startswith("-") or token in ("&&", "||", "|", ";", ">", "<", ">>"):
            continue

        # Check if token looks like a path with traversal
        if ".." in token:
            # Resolve the path relative to working_dir
            try:
                if token.startswith("/"):
                    resolved = Path(token).resolve()
                else:
                    resolved = (working_path / token).resolve()
                # Check if resolved path is within working_dir
                try:
                    resolved.relative_to(working_path)
                except ValueError:
                    detail = "Command restricted: path escapes project directory"
                    raise HTTPException(status_code=403, detail=detail) from None
            except (OSError, RuntimeError):
                # Path resolution failed, be conservative and block
                detail = "Command restricted: invalid path in command"
                raise HTTPException(status_code=403, detail=detail) from None


def validate_workdir(workdir: str | None, project_dir: str) -> None:
    """Validate that a working directory is within the project.

    Args:
        workdir: The requested working directory (may be None).
        project_dir: The project root directory.

    Raises:
        HTTPException: If workdir escapes the project directory.
    """
    if workdir is None:
        return

    project_path = Path(project_dir).resolve()
    work_path = Path(workdir).resolve()
    try:
        work_path.relative_to(project_path)
    except ValueError:
        detail = "Command restricted: working directory outside project"
        raise HTTPException(status_code=403, detail=detail) from None
