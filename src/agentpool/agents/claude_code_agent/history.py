"""Claude Code message history loader and converter.

This module provides utilities for loading Claude Code's conversation history
from ~/.claude/projects/ and converting it to pydantic-ai message format.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

import anyenv


if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai import ModelRequest, ModelResponse

from pydantic import BaseModel, Field


# Claude Code history entry types


class ClaudeCodeUsage(BaseModel):
    """Token usage information from Claude Code."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ClaudeCodeTextContent(BaseModel):
    """Text content block in Claude Code messages."""

    type: Literal["text"]
    text: str


class ClaudeCodeToolUseContent(BaseModel):
    """Tool use content block in Claude Code messages."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ClaudeCodeToolResultContent(BaseModel):
    """Tool result content block in Claude Code messages."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: list[ClaudeCodeTextContent] | str


class ClaudeCodeThinkingContent(BaseModel):
    """Thinking content block in Claude Code messages."""

    type: Literal["thinking"]
    thinking: str


ClaudeCodeContentBlock = Annotated[
    ClaudeCodeTextContent
    | ClaudeCodeToolUseContent
    | ClaudeCodeToolResultContent
    | ClaudeCodeThinkingContent,
    Field(discriminator="type"),
]


class ClaudeCodeUserMessage(BaseModel):
    """User message payload in Claude Code format."""

    role: Literal["user"]
    content: str | list[ClaudeCodeContentBlock]


class ClaudeCodeAssistantMessage(BaseModel):
    """Assistant message payload in Claude Code format."""

    model: str | None = None
    id: str | None = None
    type: Literal["message"] = "message"
    role: Literal["assistant"]
    content: list[ClaudeCodeContentBlock]
    stop_reason: str | None = None
    usage: ClaudeCodeUsage | None = None


class ClaudeCodeUserEntry(BaseModel):
    """A user entry in Claude Code's JSONL history."""

    type: Literal["user"]
    message: ClaudeCodeUserMessage
    uuid: str
    parent_uuid: str | None = Field(default=None, alias="parentUuid")
    session_id: str = Field(alias="sessionId")
    timestamp: datetime
    cwd: str | None = None
    version: str | None = None
    git_branch: str | None = Field(default=None, alias="gitBranch")
    is_sidechain: bool = Field(default=False, alias="isSidechain")
    user_type: str | None = Field(default=None, alias="userType")


class ClaudeCodeAssistantEntry(BaseModel):
    """An assistant entry in Claude Code's JSONL history."""

    type: Literal["assistant"]
    message: ClaudeCodeAssistantMessage
    uuid: str
    parent_uuid: str | None = Field(default=None, alias="parentUuid")
    session_id: str = Field(alias="sessionId")
    timestamp: datetime
    request_id: str | None = Field(default=None, alias="requestId")
    cwd: str | None = None
    version: str | None = None
    git_branch: str | None = Field(default=None, alias="gitBranch")
    is_sidechain: bool = Field(default=False, alias="isSidechain")
    user_type: str | None = Field(default=None, alias="userType")


class ClaudeCodeQueueOperation(BaseModel):
    """A queue operation entry (metadata, not a message)."""

    type: Literal["queue-operation"]
    operation: str
    timestamp: datetime
    session_id: str = Field(alias="sessionId")


class ClaudeCodeSummary(BaseModel):
    """A summary entry in Claude Code's history."""

    type: Literal["summary"]
    summary: str
    uuid: str
    parent_uuid: str | None = Field(default=None, alias="parentUuid")
    session_id: str = Field(alias="sessionId")
    timestamp: datetime
    is_sidechain: bool = Field(default=False, alias="isSidechain")


ClaudeCodeEntry = Annotated[
    ClaudeCodeUserEntry | ClaudeCodeAssistantEntry | ClaudeCodeQueueOperation | ClaudeCodeSummary,
    Field(discriminator="type"),
]

# Message entries that have uuid and parent_uuid (excludes queue operations)
ClaudeCodeMessageEntry = ClaudeCodeUserEntry | ClaudeCodeAssistantEntry | ClaudeCodeSummary


def parse_entry(line: str) -> ClaudeCodeEntry | None:
    """Parse a single JSONL line into a Claude Code entry.

    Args:
        line: A single line from the JSONL file

    Returns:
        Parsed entry or None if the line is empty or unparseable
    """
    line = line.strip()
    if not line:
        return None

    data = anyenv.load_json(line, return_type=dict)
    match data.get("type"):
        case "user":
            return ClaudeCodeUserEntry.model_validate(data)
        case "assistant":
            return ClaudeCodeAssistantEntry.model_validate(data)
        case "queue-operation":
            return ClaudeCodeQueueOperation.model_validate(data)
        case "summary":
            return ClaudeCodeSummary.model_validate(data)
        case _:
            return None


def load_session(session_path: Path) -> list[ClaudeCodeEntry]:
    """Load all entries from a Claude Code session file.

    Args:
        session_path: Path to the .jsonl session file

    Returns:
        List of parsed entries
    """
    with session_path.open() as f:
        return [entry for line in f if (entry := parse_entry(line))]


def get_main_conversation(
    entries: list[ClaudeCodeEntry],
    *,
    include_sidechains: bool = False,
) -> list[ClaudeCodeMessageEntry]:
    """Extract the main conversation thread from entries.

    Claude Code supports forking conversations via parentUuid. This function
    follows the parent chain to reconstruct the main conversation, optionally
    including or excluding sidechain messages.

    Args:
        entries: All entries from the session
        include_sidechains: If True, include sidechain entries. If False (default),
            only include the main conversation thread.

    Returns:
        Entries in conversation order, following the parent chain
    """
    # Filter to message entries (not queue operations)
    message_entries: list[ClaudeCodeMessageEntry] = [
        e
        for e in entries
        if isinstance(e, ClaudeCodeUserEntry | ClaudeCodeAssistantEntry | ClaudeCodeSummary)
    ]

    if not message_entries:
        return []

    # Build children lookup
    children: dict[str | None, list[ClaudeCodeMessageEntry]] = {}
    for entry in message_entries:
        parent = entry.parent_uuid
        children.setdefault(parent, []).append(entry)

    # Find root(s) - entries with no parent
    roots = children.get(None, [])

    if not roots:
        # No roots found, fall back to file order
        if include_sidechains:
            return message_entries
        return [e for e in message_entries if not e.is_sidechain]

    # Walk the tree, preferring non-sidechain entries
    result: list[ClaudeCodeMessageEntry] = []

    def walk(entry: ClaudeCodeMessageEntry) -> None:
        if include_sidechains or not entry.is_sidechain:
            result.append(entry)

        # Get children of this entry
        entry_children = children.get(entry.uuid, [])

        # Sort children: non-sidechains first, then by timestamp
        entry_children.sort(key=lambda e: (e.is_sidechain, e.timestamp))

        for child in entry_children:
            walk(child)

    # Start from roots (sorted by timestamp)
    roots.sort(key=lambda e: e.timestamp)
    for root in roots:
        walk(root)

    return result


def get_claude_data_dir() -> Path:
    """Get the Claude Code data directory path.

    Claude Code stores data in ~/.claude rather than the XDG data directory.
    """
    from pathlib import Path

    return Path.home() / ".claude"


def get_claude_projects_dir() -> Path:
    """Get the Claude Code projects directory path."""
    return get_claude_data_dir() / "projects"


def path_to_claude_dir_name(project_path: str) -> str:
    """Convert a filesystem path to Claude Code's directory naming format.

    Claude Code replaces '/' with '-', so '/home/user/project' becomes '-home-user-project'.

    Args:
        project_path: The filesystem path

    Returns:
        The Claude Code directory name format
    """
    return project_path.replace("/", "-")


def list_project_sessions(project_path: str) -> list[Path]:
    """List all session files for a project.

    Args:
        project_path: The project path (will be converted to Claude's format)

    Returns:
        List of session file paths, sorted by modification time (newest first)
    """
    projects_dir = get_claude_projects_dir()
    project_dir_name = path_to_claude_dir_name(project_path)
    project_dir = projects_dir / project_dir_name

    if not project_dir.exists():
        return []

    sessions = list(project_dir.glob("*.jsonl"))
    return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)


def convert_to_pydantic_ai(
    entries: list[ClaudeCodeEntry],
    *,
    include_sidechains: bool = False,
    follow_parent_chain: bool = True,
) -> list[ModelRequest | ModelResponse]:
    """Convert Claude Code entries to pydantic-ai message format.

    Args:
        entries: List of Claude Code history entries
        include_sidechains: If True, include sidechain (forked) messages
        follow_parent_chain: If True (default), reconstruct conversation order
            by following parentUuid links. If False, use file order.

    Returns:
        List of ModelRequest and ModelResponse objects
    """
    from pydantic_ai import ModelRequest, ModelResponse

    # Optionally reconstruct proper conversation order
    conversation: list[ClaudeCodeEntry] | list[ClaudeCodeMessageEntry]
    if follow_parent_chain:
        conversation = get_main_conversation(entries, include_sidechains=include_sidechains)
    else:
        conversation = entries
    from pydantic_ai.messages import (
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    messages: list[ModelRequest | ModelResponse] = []

    for entry in conversation:
        match entry:
            case ClaudeCodeUserEntry():
                parts: list[Any] = []
                metadata = {
                    "uuid": entry.uuid,
                    "timestamp": entry.timestamp.isoformat(),
                    "sessionId": entry.session_id,
                    "cwd": entry.cwd,
                    "gitBranch": entry.git_branch,
                    "isSidechain": entry.is_sidechain,
                }

                content = entry.message.content
                if isinstance(content, str):
                    parts.append(UserPromptPart(content=content))
                else:
                    for block in content:
                        match block:
                            case ClaudeCodeTextContent():
                                parts.append(UserPromptPart(content=block.text))
                            case ClaudeCodeToolResultContent():
                                # Extract text from tool result content
                                if isinstance(block.content, str):
                                    result_content = block.content
                                else:
                                    result_content = "\n".join(
                                        c.text
                                        for c in block.content
                                        if isinstance(c, ClaudeCodeTextContent)
                                    )
                                parts.append(
                                    ToolReturnPart(
                                        tool_name="",  # Not available in history
                                        content=result_content,
                                        tool_call_id=block.tool_use_id,
                                    )
                                )

                if parts:
                    messages.append(ModelRequest(parts=parts, metadata=metadata))

            case ClaudeCodeAssistantEntry():
                parts = []
                metadata = {
                    "uuid": entry.uuid,
                    "timestamp": entry.timestamp.isoformat(),
                    "sessionId": entry.session_id,
                    "requestId": entry.request_id,
                    "cwd": entry.cwd,
                    "gitBranch": entry.git_branch,
                    "isSidechain": entry.is_sidechain,
                }

                for block in entry.message.content:
                    match block:
                        case ClaudeCodeTextContent():
                            parts.append(TextPart(content=block.text))
                        case ClaudeCodeToolUseContent():
                            parts.append(
                                ToolCallPart(
                                    tool_name=block.name,
                                    args=block.input,
                                    tool_call_id=block.id,
                                )
                            )
                        case ClaudeCodeThinkingContent():
                            parts.append(ThinkingPart(content=block.thinking))

                if parts:
                    messages.append(
                        ModelResponse(
                            parts=parts,
                            model_name=entry.message.model,
                            provider_response_id=entry.message.id,
                            metadata=metadata,
                        )
                    )

            case ClaudeCodeSummary():
                # Summaries can be added as system context if needed
                metadata = {
                    "uuid": entry.uuid,
                    "timestamp": entry.timestamp.isoformat(),
                    "sessionId": entry.session_id,
                    "type": "summary",
                }
                messages.append(
                    ModelRequest(
                        parts=[UserPromptPart(content=f"[Summary]: {entry.summary}")],
                        metadata=metadata,
                    )
                )

            case ClaudeCodeQueueOperation():
                # Skip queue operations - they're metadata, not messages
                pass

    return messages


def load_session_as_pydantic_ai(session_path: Path) -> list[ModelRequest | ModelResponse]:
    """Load a Claude Code session and convert to pydantic-ai format.

    Args:
        session_path: Path to the .jsonl session file

    Returns:
        List of ModelRequest and ModelResponse objects
    """
    entries = load_session(session_path)
    return convert_to_pydantic_ai(entries)


def get_latest_session(project_path: str) -> Path | None:
    """Get the most recent session file for a project.

    Args:
        project_path: The project path

    Returns:
        Path to the latest session file, or None if no sessions exist
    """
    sessions = list_project_sessions(project_path)
    return sessions[0] if sessions else None
