"""Claude Code storage provider.

This module implements storage compatible with Claude Code's filesystem format,
enabling interoperability between agentpool and Claude Code.

Key features:
- JSONL-based conversation logs per project
- Multi-agent support (main + sub-agents)
- Message ancestry tracking
- Conversation forking and branching

See ARCHITECTURE.md for detailed documentation of the storage format.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from agentpool.log import get_logger


logger = get_logger(__name__)


StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
ContentType = Literal["text", "tool_use", "tool_result", "thinking"]
MessageType = Literal[
    "user", "assistant", "queue-operation", "system", "summary", "file-history-snapshot"
]
UserType = Literal["external", "internal"]


class ClaudeBaseModel(BaseModel):
    """Base class for Claude history models."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ClaudeUsage(BaseModel):
    """Token usage from Claude API response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ClaudeMessageContent(BaseModel):
    """Content block in Claude message.

    Supports: text, tool_use, tool_result, thinking blocks.
    """

    type: ContentType
    # For text blocks
    text: str | None = None
    # For tool_use blocks
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    # For tool_result blocks
    tool_use_id: str | None = None
    content: list[dict[str, Any]] | str | None = None  # Can be array or string
    is_error: bool | None = None
    # For thinking blocks
    thinking: str | None = None
    signature: str | None = None


class ClaudeApiMessage(BaseModel):
    """Claude API message structure."""

    model: str
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"]
    content: str | list[ClaudeMessageContent]
    stop_reason: StopReason = None
    usage: ClaudeUsage = Field(default_factory=ClaudeUsage)


class ClaudeUserMessage(BaseModel):
    """User message content."""

    role: Literal["user"]
    content: str | list[ClaudeMessageContent]


class ClaudeMessageEntryBase(ClaudeBaseModel):
    """Base for user/assistant message entries."""

    uuid: str
    parent_uuid: str | None = None
    session_id: str = Field(alias="sessionId")
    timestamp: str
    message: ClaudeApiMessage | ClaudeUserMessage

    # Context (NOT USED directly)
    cwd: str = ""
    git_branch: str = ""
    version: str = ""

    # Metadata (NOT USED)
    user_type: UserType = "external"
    is_sidechain: bool = False
    request_id: str | None = None
    agent_id: str | None = None
    # toolUseResult can be list, dict, or string (error message)
    tool_use_result: list[dict[str, Any]] | dict[str, Any] | str | None = None


class ClaudeUserEntry(ClaudeMessageEntryBase):
    """User message entry."""

    type: Literal["user"]


class ClaudeAssistantEntry(ClaudeMessageEntryBase):
    """Assistant message entry."""

    type: Literal["assistant"]


class ClaudeQueueOperationEntry(ClaudeBaseModel):
    """Queue operation entry (not a message)."""

    type: Literal["queue-operation"]
    session_id: str
    timestamp: str
    operation: str


class ClaudeSystemEntry(ClaudeBaseModel):
    """System message entry (context, prompts, etc.)."""

    type: Literal["system"]
    uuid: str
    parent_uuid: str | None = None
    session_id: str
    timestamp: str
    content: str | None = None  # Optional for subtypes like turn_duration
    subtype: str | None = None
    duration_ms: int | None = None  # For turn_duration subtype
    slug: str | None = None
    level: int | str | None = None
    is_meta: bool = False
    logical_parent_uuid: str | None = None
    compact_metadata: dict[str, Any] | None = None
    # Common fields
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    user_type: UserType = "external"
    is_sidechain: bool = False


class ClaudeSummaryEntry(ClaudeBaseModel):
    """Summary entry (conversation summary)."""

    type: Literal["summary"]
    leaf_uuid: str
    summary: str


class ClaudeFileHistoryEntry(ClaudeBaseModel):
    """File history snapshot entry."""

    type: Literal["file-history-snapshot"]
    message_id: str
    snapshot: dict[str, Any]
    is_snapshot_update: bool = False


class ClaudeMcpProgressData(ClaudeBaseModel):
    """Progress data for MCP tool operations."""

    type: Literal["mcp_progress"]
    status: Literal["started", "completed", "failed"] | None = None
    server_name: str | None = None
    tool_name: str | None = None
    elapsed_time_ms: int | None = None


class ClaudeBashProgressData(ClaudeBaseModel):
    """Progress data for MCP tool operations."""

    type: Literal["bash_progress"]
    output: str | None = None
    full_output: str | None = None
    elapsed_time_seconds: int | None = None
    total_lines: int | None = None


class ClaudeHookProgressData(ClaudeBaseModel):
    """Progress data for hook operations."""

    type: Literal["hook_progress"]
    hook_event: str | None = None
    hook_name: str | None = None
    command: str | None = None


class ClaudeWaitingForTaskData(ClaudeBaseModel):
    """Progress data for waiting task operations."""

    type: Literal["waiting_for_task"]
    task_description: str | None = None
    task_type: str | None = None


ClaudeProgressData = Annotated[
    ClaudeMcpProgressData
    | ClaudeBashProgressData
    | ClaudeHookProgressData
    | ClaudeWaitingForTaskData,
    Field(discriminator="type"),
]


class ClaudeProgressEntry(ClaudeBaseModel):
    """Progress entry for tracking tool execution status."""

    type: Literal["progress"]
    uuid: str
    slug: str | None = None
    parent_uuid: str | None = None
    session_id: str
    timestamp: str
    data: ClaudeProgressData
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None
    # Common fields
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    user_type: UserType = "external"
    is_sidechain: bool = False


# Discriminated union for all entry types
ClaudeJSONLEntry = Annotated[
    ClaudeUserEntry
    | ClaudeAssistantEntry
    | ClaudeQueueOperationEntry
    | ClaudeSystemEntry
    | ClaudeSummaryEntry
    | ClaudeFileHistoryEntry
    | ClaudeProgressEntry,
    Field(discriminator="type"),
]
