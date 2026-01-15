"""Pydantic models for Codex JSON-RPC API requests and responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from codex_adapter.codex_types import ModelProvider


# ============================================================================
# Base classes with shared configuration
# ============================================================================


class CodexBaseModel(BaseModel):
    """Base model for all Codex API models.

    Provides:
    - Strict validation (forbids extra fields)
    - Snake_case Python fields with camelCase JSON aliases
    - Both field names and aliases accepted for parsing (populate_by_name=True)
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=to_camel,
    )


# ============================================================================
# Request parameter models
# ============================================================================


class ClientInfo(CodexBaseModel):
    """Client information for initialization."""

    name: str
    version: str


class InitializeParams(CodexBaseModel):
    """Parameters for initialize request."""

    client_info: ClientInfo


class ThreadStartParams(CodexBaseModel):
    """Parameters for thread/start request."""

    cwd: str | None = None
    model: str | None = None
    effort: Literal["low", "medium", "high"] | None = None


class ThreadResumeParams(CodexBaseModel):
    """Parameters for thread/resume request."""

    thread_id: str


class ThreadForkParams(CodexBaseModel):
    """Parameters for thread/fork request."""

    thread_id: str


class ThreadListParams(CodexBaseModel):
    """Parameters for thread/list request."""

    cursor: str | None = None
    limit: int | None = None
    model_providers: list[str] | None = None


class ThreadArchiveParams(CodexBaseModel):
    """Parameters for thread/archive request."""

    thread_id: str


class ThreadRollbackParams(CodexBaseModel):
    """Parameters for thread/rollback request."""

    thread_id: str
    turns: int


class TextInputItem(CodexBaseModel):
    """Text input for a turn."""

    type: Literal["text"] = "text"
    text: str


class LocalImageInputItem(CodexBaseModel):
    """Local image file input for a turn."""

    type: Literal["local_image"] = "local_image"
    path: str


# Discriminated union of input types
TurnInputItem = TextInputItem | LocalImageInputItem


class TurnStartParams(CodexBaseModel):
    """Parameters for turn/start request."""

    thread_id: str
    input: list[TurnInputItem]
    model: str | None = None
    effort: Literal["low", "medium", "high"] | None = None
    approval_policy: Literal["always", "never", "auto"] | None = None
    output_schema: dict[str, Any] | None = None  # JSON Schema - arbitrary structure


class TurnInterruptParams(CodexBaseModel):
    """Parameters for turn/interrupt request."""

    thread_id: str
    turn_id: str


class SkillsListParams(CodexBaseModel):
    """Parameters for skills/list request."""

    cwd: str | None = None
    force_reload: bool = False


class CommandExecParams(CodexBaseModel):
    """Parameters for command/exec request."""

    command: list[str]
    cwd: str | None = None
    sandbox_policy: dict[str, Any] | None = None  # Sandbox config - flexible structure
    timeout_ms: int | None = None


# ============================================================================
# Response models
# ============================================================================


class GitInfo(CodexBaseModel):
    """Git metadata captured when thread was created."""

    sha: str | None = None
    branch: str | None = None
    origin_url: str | None = None


class TurnStatus(CodexBaseModel):
    """Turn status enumeration."""

    # This is actually an enum in Rust but sent as string
    status: Literal["completed", "interrupted", "failed", "inProgress"]


class TurnError(CodexBaseModel):
    """Turn error information."""

    message: str
    codex_error_info: dict[str, Any] | str | None = (
        None  # Error metadata - varied structure (dict or string like "other")
    )
    additional_details: str | None = None


# ============================================================================
# UserInput and dependent types for ThreadItem
# ============================================================================


class UserInputText(CodexBaseModel):
    """Text user input."""

    type: Literal["text"] = "text"
    text: str


class UserInputImage(CodexBaseModel):
    """Image URL user input."""

    type: Literal["image"] = "image"
    url: str


class UserInputLocalImage(CodexBaseModel):
    """Local image file user input."""

    type: Literal["localImage"] = "localImage"
    path: str


class UserInputSkill(CodexBaseModel):
    """Skill file user input."""

    type: Literal["skill"] = "skill"
    name: str
    path: str


# Discriminated union of user input types
UserInput = UserInputText | UserInputImage | UserInputLocalImage | UserInputSkill


class CommandActionRead(CodexBaseModel):
    """Read command action."""

    type: Literal["read"] = "read"
    command: str
    name: str
    path: str


class CommandActionListFiles(CodexBaseModel):
    """List files command action."""

    type: Literal["listFiles"] = "listFiles"
    command: str
    path: str | None = None


class CommandActionSearch(CodexBaseModel):
    """Search command action."""

    type: Literal["search"] = "search"
    command: str
    query: str | None = None
    path: str | None = None


class CommandActionUnknown(CodexBaseModel):
    """Unknown command action."""

    type: Literal["unknown"] = "unknown"
    command: str


# Discriminated union of command actions
CommandAction = (
    CommandActionRead | CommandActionListFiles | CommandActionSearch | CommandActionUnknown
)


class FileUpdateChange(CodexBaseModel):
    """File update change."""

    path: str
    kind: Literal["add", "delete", "update"]
    move_path: str | None = None
    diff: str


class McpContentBlock(CodexBaseModel):
    """MCP content block (from external mcp_types crate).

    We allow extra fields since this comes from an external library.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=to_camel,
    )


class McpToolCallResult(CodexBaseModel):
    """MCP tool call result."""

    content: list[McpContentBlock]
    structured_content: dict[str, Any] | list[Any] | str | int | float | bool | None = None


class McpToolCallError(CodexBaseModel):
    """MCP tool call error."""

    message: str


# ============================================================================
# ThreadItem discriminated union - 10 variants
# ============================================================================


class ThreadItemUserMessage(CodexBaseModel):
    """User message item."""

    type: Literal["userMessage"] = "userMessage"
    id: str
    content: list[UserInput]


class ThreadItemAgentMessage(CodexBaseModel):
    """Agent message item."""

    type: Literal["agentMessage"] = "agentMessage"
    id: str
    text: str


class ThreadItemReasoning(CodexBaseModel):
    """Reasoning item."""

    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[str] = Field(default_factory=list)
    content: list[str] = Field(default_factory=list)


class ThreadItemCommandExecution(CodexBaseModel):
    """Command execution item."""

    type: Literal["commandExecution"] = "commandExecution"
    id: str
    command: str
    cwd: str
    process_id: str | None = None
    status: Literal["inProgress", "completed", "failed", "declined"]
    command_actions: list[CommandAction] = Field(default_factory=list)
    aggregated_output: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None


class ThreadItemFileChange(CodexBaseModel):
    """File change item."""

    type: Literal["fileChange"] = "fileChange"
    id: str
    changes: list[FileUpdateChange]
    status: Literal["inProgress", "completed", "failed", "declined"]


class ThreadItemMcpToolCall(CodexBaseModel):
    """MCP tool call item."""

    type: Literal["mcpToolCall"] = "mcpToolCall"
    id: str
    server: str
    tool: str
    status: Literal["inProgress", "completed", "failed"]
    arguments: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    result: McpToolCallResult | None = None
    error: McpToolCallError | None = None
    duration_ms: int | None = None


class ThreadItemWebSearch(CodexBaseModel):
    """Web search item."""

    type: Literal["webSearch"] = "webSearch"
    id: str
    query: str


class ThreadItemImageView(CodexBaseModel):
    """Image view item."""

    type: Literal["imageView"] = "imageView"
    id: str
    path: str


class ThreadItemEnteredReviewMode(CodexBaseModel):
    """Entered review mode item."""

    type: Literal["enteredReviewMode"] = "enteredReviewMode"
    id: str
    review: str


class ThreadItemExitedReviewMode(CodexBaseModel):
    """Exited review mode item."""

    type: Literal["exitedReviewMode"] = "exitedReviewMode"
    id: str
    review: str


# Discriminated union of all ThreadItem types
ThreadItem = (
    ThreadItemUserMessage
    | ThreadItemAgentMessage
    | ThreadItemReasoning
    | ThreadItemCommandExecution
    | ThreadItemFileChange
    | ThreadItemMcpToolCall
    | ThreadItemWebSearch
    | ThreadItemImageView
    | ThreadItemEnteredReviewMode
    | ThreadItemExitedReviewMode
)


class Turn(CodexBaseModel):
    """Turn data structure."""

    id: str
    items: list[ThreadItem] = Field(default_factory=list)
    status: Literal["completed", "interrupted", "failed", "inProgress"] = "inProgress"
    error: TurnError | None = None


class Thread(CodexBaseModel):
    """Thread data structure."""

    id: str
    preview: str = ""
    model_provider: str = "openai"
    created_at: int = 0
    path: str = ""
    cwd: str = ""
    cli_version: str = ""
    source: str = "appServer"
    git_info: GitInfo | None = None
    turns: list[Turn] = Field(default_factory=list)


class ThreadData(CodexBaseModel):
    """Thread data in responses (legacy, kept for compatibility)."""

    id: str
    preview: str = ""
    model_provider: ModelProvider = "openai"
    created_at: int = 0
    path: str | None = None
    cwd: str | None = None
    cli_version: str | None = None
    source: str | None = None
    git_info: GitInfo | None = None
    turns: list[Turn] = Field(default_factory=list)


class ThreadResponse(CodexBaseModel):
    """Response for thread operations."""

    thread: ThreadData
    model: str | None = None
    model_provider: ModelProvider | None = None
    cwd: str | None = None
    approval_policy: str | None = None
    sandbox: dict[str, Any] | None = None  # Sandbox config - flexible structure
    reasoning_effort: str | None = None


class TurnData(CodexBaseModel):
    """Turn data in responses (legacy, kept for compatibility)."""

    id: str
    status: Literal["pending", "inProgress", "completed", "error", "interrupted"] = "pending"
    thread_id: str | None = None
    items: list[ThreadItem] = Field(default_factory=list)
    error: str | None = None


class TurnStartResponse(CodexBaseModel):
    """Response for turn/start request."""

    turn: TurnData


class ThreadListResponse(CodexBaseModel):
    """Response for thread/list request."""

    data: list[ThreadData]
    next_cursor: str | None = None


class ThreadLoadedListResponse(CodexBaseModel):
    """Response for thread/loaded/list request."""

    data: list[str]


class ThreadRollbackResponse(CodexBaseModel):
    """Response for thread/rollback request."""

    thread: ThreadData
    turns: list[Turn]


class SkillData(CodexBaseModel):
    """A single skill definition."""

    name: str
    description: str | None = None


class SkillsContainer(CodexBaseModel):
    """Container for skills with cwd."""

    cwd: str
    skills: list[SkillData]
    errors: list[str] = Field(default_factory=list)


class SkillsListResponse(CodexBaseModel):
    """Response for skills/list request."""

    data: list[SkillsContainer]


class ModelData(CodexBaseModel):
    """A single model definition."""

    id: str
    model: str


class ModelListResponse(CodexBaseModel):
    """Response for model/list request."""

    data: list[ModelData]


class CommandExecResponse(CodexBaseModel):
    """Response for command/exec request."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""


# ============================================================================
# JSON-RPC message models
# ============================================================================


class JsonRpcRequest(CodexBaseModel):
    """JSON-RPC 2.0 request message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)  # Method-specific params


class JsonRpcError(CodexBaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None


class JsonRpcResponse(CodexBaseModel):
    """JSON-RPC 2.0 response message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int
    result: Any = None
    error: JsonRpcError | None = None


class JsonRpcNotification(CodexBaseModel):
    """JSON-RPC 2.0 notification message (no id)."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None  # Event-specific params


# ============================================================================
# Event payload models - exact match to app-server-protocol v2
# ============================================================================


class TokenUsageBreakdown(CodexBaseModel):
    """Token usage breakdown."""

    total_tokens: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int = 0


class ThreadTokenUsage(CodexBaseModel):
    """Thread token usage information."""

    total: TokenUsageBreakdown
    last: TokenUsageBreakdown
    model_context_window: int | None = None


class Usage(CodexBaseModel):
    """Simple token usage (legacy)."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


# Thread lifecycle notifications


class ThreadStartedData(CodexBaseModel):
    """Payload for thread/started notification (V2 protocol)."""

    thread: Thread
    thread_id: str | None = None


class ThreadTokenUsageUpdatedData(CodexBaseModel):
    """Payload for thread/tokenUsage/updated notification (V2 protocol)."""

    thread_id: str
    turn_id: str
    token_usage: ThreadTokenUsage


class ThreadCompactedData(CodexBaseModel):
    """Payload for thread/compacted notification."""

    thread_id: str


# Turn lifecycle notifications


class TurnStartedData(CodexBaseModel):
    """Payload for turn/started notification (V2 protocol)."""

    thread_id: str
    turn: Turn


class TurnCompletedData(CodexBaseModel):
    """Payload for turn/completed notification (V2 protocol)."""

    thread_id: str
    turn: Turn


class TurnErrorData(CodexBaseModel):
    """Payload for turn/error notification."""

    thread_id: str
    turn_id: str
    error: str


class TurnDiffUpdatedData(CodexBaseModel):
    """Payload for turn/diff/updated notification."""

    thread_id: str
    turn_id: str
    diff: str


class TurnPlanStep(CodexBaseModel):
    """A single step in a turn plan."""

    step: str
    status: Literal["pending", "inProgress", "completed"]


class TurnPlanUpdatedData(CodexBaseModel):
    """Payload for turn/plan/updated notification."""

    thread_id: str
    turn_id: str
    explanation: str | None = None
    plan: list[TurnPlanStep]


# Item lifecycle notifications


class ItemStartedData(CodexBaseModel):
    """Payload for item/started notification (V2 protocol)."""

    thread_id: str
    turn_id: str
    item: ThreadItem


class ItemCompletedData(CodexBaseModel):
    """Payload for item/completed notification (V2 protocol)."""

    thread_id: str
    turn_id: str
    item: ThreadItem


class RawResponseItemCompletedData(CodexBaseModel):
    """Payload for rawResponseItem/completed notification."""

    thread_id: str
    turn_id: str
    item: ThreadItem


# Item delta notifications


class AgentMessageDeltaData(CodexBaseModel):
    """Payload for item/agentMessage/delta notification."""

    thread_id: str
    turn_id: str
    item_id: str
    delta: str


class ReasoningTextDeltaData(CodexBaseModel):
    """Payload for item/reasoning/textDelta notification."""

    thread_id: str
    turn_id: str
    item_id: str
    delta: str
    content_index: int


class ReasoningSummaryTextDeltaData(CodexBaseModel):
    """Payload for item/reasoning/summaryTextDelta notification."""

    thread_id: str
    turn_id: str
    item_id: str
    delta: str
    summary_index: int


class ReasoningSummaryPartAddedData(CodexBaseModel):
    """Payload for item/reasoning/summaryPartAdded notification."""

    thread_id: str
    turn_id: str
    item_id: str
    summary_index: int


class CommandExecutionOutputDeltaData(CodexBaseModel):
    """Payload for item/commandExecution/outputDelta notification."""

    thread_id: str
    turn_id: str
    item_id: str
    delta: str


class CommandExecutionTerminalInteractionData(CodexBaseModel):
    """Payload for item/commandExecution/terminalInteraction notification."""

    thread_id: str
    turn_id: str
    item_id: str
    process_id: str
    stdin: str


class FileChangeOutputDeltaData(CodexBaseModel):
    """Payload for item/fileChange/outputDelta notification."""

    thread_id: str
    turn_id: str
    item_id: str
    delta: str


class McpToolCallProgressData(CodexBaseModel):
    """Payload for item/mcpToolCall/progress notification."""

    thread_id: str
    turn_id: str
    item_id: str
    message: str


# MCP/Account/System notifications


class McpServerOAuthLoginCompletedData(CodexBaseModel):
    """Payload for mcpServer/oauthLogin/completed notification."""

    name: str
    success: bool
    error: str | None = None


class AccountUpdatedData(CodexBaseModel):
    """Payload for account/updated notification."""

    auth_mode: str | None = None


class RateLimitWindow(CodexBaseModel):
    """Rate limit window information."""

    used_percent: int
    window_duration_mins: int | None = None
    resets_at: int | None = None


class CreditsSnapshot(CodexBaseModel):
    """Credits snapshot information."""

    has_credits: bool
    unlimited: bool
    balance: str | None = None


class RateLimitSnapshot(CodexBaseModel):
    """Rate limit snapshot."""

    primary: RateLimitWindow | None = None
    secondary: RateLimitWindow | None = None
    credits: CreditsSnapshot | None = None
    plan_type: str | None = None


class AccountRateLimitsUpdatedData(CodexBaseModel):
    """Payload for account/rateLimits/updated notification."""

    rate_limits: RateLimitSnapshot


class AccountLoginCompletedData(CodexBaseModel):
    """Payload for account/login/completed notification."""

    login_id: str | None = None
    success: bool
    error: str | None = None


class AuthStatusChangeData(CodexBaseModel):
    """Payload for authStatusChange notification (legacy v1)."""

    status: str


class LoginChatGptCompleteData(CodexBaseModel):
    """Payload for loginChatGptComplete notification (legacy v1)."""

    success: bool


class SessionConfiguredData(CodexBaseModel):
    """Payload for sessionConfigured notification."""

    config: dict[str, Any]  # Session config - flexible structure


class DeprecationNoticeData(CodexBaseModel):
    """Payload for deprecationNotice notification."""

    summary: str
    details: str | None = None


class WindowsWorldWritableWarningData(CodexBaseModel):
    """Payload for windows/worldWritableWarning notification."""

    sample_paths: list[str]
    extra_count: int
    failed_scan: bool


class ErrorEventData(CodexBaseModel):
    """Payload for error event."""

    error: TurnError
    will_retry: bool
    thread_id: str
    turn_id: str


class GenericEventData(CodexBaseModel):
    """Generic payload for events we don't have specific types for yet.

    Allows any extra fields since we don't know the exact structure.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=to_camel,
    )

    thread_id: str | None = None
    turn_id: str | None = None
    item_id: str | None = None


# Union type of all event data
EventData = (
    # Thread lifecycle
    ThreadStartedData
    | ThreadTokenUsageUpdatedData
    | ThreadCompactedData
    # Turn lifecycle
    | TurnStartedData
    | TurnCompletedData
    | TurnErrorData
    | TurnDiffUpdatedData
    | TurnPlanUpdatedData
    # Item lifecycle
    | ItemStartedData
    | ItemCompletedData
    | RawResponseItemCompletedData
    # Item deltas - agent messages
    | AgentMessageDeltaData
    # Item deltas - reasoning
    | ReasoningTextDeltaData
    | ReasoningSummaryTextDeltaData
    | ReasoningSummaryPartAddedData
    # Item deltas - command execution
    | CommandExecutionOutputDeltaData
    | CommandExecutionTerminalInteractionData
    # Item deltas - file changes
    | FileChangeOutputDeltaData
    # Item deltas - MCP tool calls
    | McpToolCallProgressData
    # MCP OAuth
    | McpServerOAuthLoginCompletedData
    # Account/Auth events
    | AccountUpdatedData
    | AccountRateLimitsUpdatedData
    | AccountLoginCompletedData
    | AuthStatusChangeData
    | LoginChatGptCompleteData
    # System events
    | SessionConfiguredData
    | DeprecationNoticeData
    | WindowsWorldWritableWarningData
    # Error events
    | ErrorEventData
    # Fallback for unknown events
    | GenericEventData
)
