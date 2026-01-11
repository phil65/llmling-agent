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
    output_schema: dict[str, Any] | None = None


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
    sandbox_policy: dict[str, Any] | None = None
    timeout_ms: int | None = None


# ============================================================================
# Response models
# ============================================================================


class ThreadData(CodexBaseModel):
    """Thread data in responses."""

    id: str
    preview: str = ""
    model_provider: ModelProvider = "openai"
    created_at: int = 0


class ThreadResponse(CodexBaseModel):
    """Response for thread operations."""

    thread: ThreadData


class TurnData(CodexBaseModel):
    """Turn data in responses."""

    id: str
    status: Literal["pending", "running", "completed", "error", "interrupted"] = "pending"
    thread_id: str | None = None


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
    turns: list[dict[str, Any]]  # Could be more specific if needed


class SkillData(CodexBaseModel):
    """A single skill definition."""

    name: str
    description: str | None = None
    # Add more fields as needed


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
    # Add more fields as needed (e.g., effort_levels, capabilities)


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
    params: dict[str, Any] = Field(default_factory=dict)


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
    params: dict[str, Any] | None = None
