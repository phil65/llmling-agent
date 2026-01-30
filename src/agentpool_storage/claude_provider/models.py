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
    session_id: str
    timestamp: str
    message: ClaudeApiMessage | ClaudeUserMessage

    # Context
    cwd: str = ""
    git_branch: str | None = None
    version: str = ""

    # Metadata
    user_type: UserType = "external"
    is_sidechain: bool = False
    is_meta: bool | None = None
    is_compact_summary: bool | None = None
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
    is_api_error_message: bool | None = None


ClaudeEntry = ClaudeUserEntry | ClaudeAssistantEntry


# Queue operation content types
class ClaudeTextContent(ClaudeBaseModel):
    """Text content block."""

    type: Literal["text"]
    text: str


class ClaudeImageSource(ClaudeBaseModel):
    """Image source with base64 data."""

    type: Literal["base64"]
    data: str
    media_type: str


class ClaudeImageContent(ClaudeBaseModel):
    """Image content block."""

    type: Literal["image"]
    source: ClaudeImageSource


class ClaudeDocumentContent(ClaudeBaseModel):
    """Document content block."""

    type: Literal["document"]
    source: ClaudeImageSource  # Same structure as image


class ClaudeToolResultContent(ClaudeBaseModel):
    """Tool result content block."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


# Union of queue operation content types
ClaudeQueueContent = (
    str | ClaudeTextContent | ClaudeImageContent | ClaudeDocumentContent | ClaudeToolResultContent
)


class ClaudeEnqueueOperation(ClaudeBaseModel):
    """Enqueue operation with content."""

    type: Literal["queue-operation"]
    operation: Literal["enqueue"]
    content: str | list[str | ClaudeQueueContent]
    session_id: str
    timestamp: str


class ClaudeDequeueOperation(ClaudeBaseModel):
    """Dequeue operation."""

    type: Literal["queue-operation"]
    operation: Literal["dequeue"]
    session_id: str
    timestamp: str


class ClaudeRemoveOperation(ClaudeBaseModel):
    """Remove operation."""

    type: Literal["queue-operation"]
    operation: Literal["remove"]
    session_id: str
    timestamp: str


class ClaudePopAllOperation(ClaudeBaseModel):
    """PopAll operation - clears all queued content."""

    type: Literal["queue-operation"]
    operation: Literal["popAll"]
    content: str
    session_id: str
    timestamp: str


# Union of queue operation types
ClaudeQueueOperationEntry = Annotated[
    ClaudeEnqueueOperation | ClaudeDequeueOperation | ClaudeRemoveOperation | ClaudePopAllOperation,
    Field(discriminator="operation"),
]


# Hook info for stop_hook_summary
class ClaudeHookInfo(ClaudeBaseModel):
    """Hook info in stop_hook_summary."""

    command: str


class ClaudeSystemEntryBase(ClaudeBaseModel):
    """Base fields for system entries."""

    type: Literal["system"]
    uuid: str
    parent_uuid: str | None = None
    session_id: str
    timestamp: str
    # Common fields
    cwd: str = ""
    git_branch: str | None = None
    version: str = ""
    user_type: UserType = "external"
    is_sidechain: bool = False
    is_meta: bool | None = None
    is_compact_summary: bool | None = None
    tool_use_result: list[dict[str, Any]] | dict[str, Any] | str | None = None


class ClaudeSystemEntryWithContent(ClaudeSystemEntryBase):
    """System entry with content (original format)."""

    content: str
    tool_use_id: str | None = Field(default=None, alias="toolUseID")
    level: Literal["info"] | None = None
    subtype: None = None


class ClaudeStopHookSummaryEntry(ClaudeSystemEntryBase):
    """Stop hook summary entry (Claude Code v2.0.76+)."""

    subtype: Literal["stop_hook_summary"]
    tool_use_id: str | None = Field(default=None, alias="toolUseID")
    level: Literal["info", "suggestion"] | None = None
    slug: str | None = None
    hook_count: int | None = None
    hook_infos: list[ClaudeHookInfo] | None = None
    hook_errors: list[Any] | None = None
    prevented_continuation: bool | None = None
    stop_reason: str | None = None
    has_output: bool | None = None


class ClaudeLocalCommandEntry(ClaudeSystemEntryBase):
    """Local command entry (e.g., /mcp, /help commands)."""

    subtype: Literal["local_command"]
    content: str
    level: Literal["info"] | None = None


class ClaudeTurnDurationEntry(ClaudeSystemEntryBase):
    """Turn duration entry."""

    subtype: Literal["turn_duration"]
    duration_ms: int | None = None


class ClaudeGenericSystemEntry(ClaudeSystemEntryBase):
    """Generic system entry for other subtypes."""

    content: str | None = None
    subtype: str | None = None
    duration_ms: int | None = None
    slug: str | None = None
    level: Literal["info", "suggestion"] | str | None = None  # noqa: PYI051
    logical_parent_uuid: str | None = None
    compact_metadata: dict[str, Any] | None = None
    tool_use_id: str | None = Field(default=None, alias="toolUseID")


# Note: We use the generic entry for parsing since Claude Code's system entries
# have many variations. For strict validation, use the specific entry types.
ClaudeSystemEntry = ClaudeGenericSystemEntry


class ClaudeSummaryEntry(ClaudeBaseModel):
    """Summary entry (conversation summary)."""

    type: Literal["summary"]
    leaf_uuid: str
    summary: str


class ClaudeFileHistorySnapshot(ClaudeBaseModel):
    """Snapshot data in file history entry."""

    message_id: str
    tracked_file_backups: dict[str, Any]
    timestamp: str


class ClaudeFileHistoryEntry(ClaudeBaseModel):
    """File history snapshot entry."""

    type: Literal["file-history-snapshot"]
    message_id: str
    snapshot: ClaudeFileHistorySnapshot | dict[str, Any]  # Allow dict for flexibility
    is_snapshot_update: bool = False


class ClaudeMcpProgressData(ClaudeBaseModel):
    """Progress data for MCP tool operations."""

    type: Literal["mcp_progress"]
    status: Literal["started", "completed", "failed"] | None = None
    server_name: str | None = None
    tool_name: str | None = None
    elapsed_time_ms: int | None = None


class ClaudeBashProgressData(ClaudeBaseModel):
    """Progress data for bash tool operations."""

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
    # Added in v2.1.12+
    prompt_text: str | None = None
    status_message: str | None = None


class ClaudeWaitingForTaskData(ClaudeBaseModel):
    """Progress data for waiting task operations."""

    type: Literal["waiting_for_task"]
    task_description: str | None = None
    task_type: str | None = None


class ClaudeAgentProgressData(ClaudeBaseModel):
    """Progress data for agent/subagent operations."""

    type: Literal["agent_progress"]
    prompt: str | None = None
    agent_id: str | None = Field(default=None, alias="agentId")
    message: dict[str, Any] | None = None
    normalized_messages: list[dict[str, Any]] | None = None
    resume: dict[str, Any] | None = None


class ClaudeQueryUpdateData(ClaudeBaseModel):
    """Progress data for search query updates."""

    type: Literal["query_update"]
    query: str


class ClaudeSearchResultsReceivedData(ClaudeBaseModel):
    """Progress data for search results received."""

    type: Literal["search_results_received"]
    result_count: int
    query: str


class ClaudeSkillProgressData(ClaudeBaseModel):
    """Progress data for skill operations."""

    type: Literal["skill_progress"]
    prompt: str | None = None
    agent_id: str | None = Field(default=None, alias="agentId")


class ClaudeTaskProgressData(ClaudeBaseModel):
    """Progress data for task tracking."""

    type: Literal["task_progress"]
    task_id: str | None = None
    task_type: str | None = None
    message: str | None = None


class ClaudeToolProgressData(ClaudeBaseModel):
    """Progress data for tool execution (SDK format)."""

    type: Literal["tool_progress"]
    tool_use_id: str | None = None
    tool_name: str | None = None
    parent_tool_use_id: str | None = None
    elapsed_time_seconds: float | None = None
    session_id: str | None = None


ClaudeProgressData = Annotated[
    ClaudeMcpProgressData
    | ClaudeBashProgressData
    | ClaudeHookProgressData
    | ClaudeWaitingForTaskData
    | ClaudeAgentProgressData
    | ClaudeQueryUpdateData
    | ClaudeSearchResultsReceivedData
    | ClaudeSkillProgressData
    | ClaudeTaskProgressData
    | ClaudeToolProgressData,
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
    agent_id: str | None = Field(default=None, alias="agentId")
    # Common fields
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    user_type: UserType = "external"
    is_sidechain: bool = False


# Discriminated union for all entry types
# Uses nested discriminator: first by "type", then queue-operation by "operation"
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
