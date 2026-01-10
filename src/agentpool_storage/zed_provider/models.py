"""Zed IDE storage format models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class ZedMentionUri(BaseModel):
    """Mention URI - can be File, Directory, Symbol, etc."""

    File: dict[str, Any] | None = None
    Directory: dict[str, Any] | None = None
    Symbol: dict[str, Any] | None = None
    Selection: dict[str, Any] | None = None
    Thread: dict[str, Any] | None = None
    TextThread: dict[str, Any] | None = None
    Rule: dict[str, Any] | None = None
    Fetch: dict[str, Any] | None = None
    PastedImage: bool | None = None


class ZedMention(BaseModel):
    """A file/symbol mention in Zed."""

    uri: ZedMentionUri
    content: str


class ZedImage(BaseModel):
    """An image in Zed (base64 encoded)."""

    source: str  # base64 encoded


class ZedThinking(BaseModel):
    """Thinking block from model."""

    text: str
    signature: str | None = None


class ZedToolUse(BaseModel):
    """Tool use block."""

    id: str
    name: str
    raw_input: str
    input: dict[str, Any]
    is_input_complete: bool = True
    thought_signature: str | None = None


class ZedToolResult(BaseModel):
    """Tool result."""

    tool_use_id: str
    tool_name: str
    is_error: bool = False
    content: dict[str, Any] | str | None = None
    output: dict[str, Any] | str | None = None


class ZedUserMessage(BaseModel):
    """User message in Zed thread."""

    id: str
    content: list[dict[str, Any]]  # Can contain Text, Image, Mention


class ZedAgentMessage(BaseModel):
    """Agent message in Zed thread."""

    content: list[dict[str, Any]]  # Can contain Text, Thinking, ToolUse
    tool_results: dict[str, ZedToolResult] = Field(default_factory=dict)
    reasoning_details: Any | None = None


class ZedMessage(BaseModel):
    """A message in Zed thread - either User or Agent."""

    User: ZedUserMessage | None = None
    Agent: ZedAgentMessage | None = None


class ZedLanguageModel(BaseModel):
    """Model configuration."""

    provider: str
    model: str


class ZedWorktreeSnapshot(BaseModel):
    """Git worktree snapshot."""

    worktree_path: str
    git_state: dict[str, Any] | None = None


class ZedProjectSnapshot(BaseModel):
    """Project snapshot with git state."""

    worktree_snapshots: list[ZedWorktreeSnapshot] = Field(default_factory=list)
    unsaved_buffer_paths: list[str] = Field(default_factory=list)
    timestamp: str | None = None


class ZedThread(BaseModel):
    """A Zed conversation thread."""

    model_config = {"populate_by_name": True}

    # v0.3.0 uses "title", v0.2.0 uses "summary"
    title: str = Field(alias="title", validation_alias=AliasChoices("title", "summary"))
    messages: list[ZedMessage | Literal["Resume"]]  # Control messages
    updated_at: str
    version: str | None = None
    detailed_summary: str | None = None  # v0.3.0 field
    detailed_summary_state: str | dict[str, Any] | None = None  # v0.2.0 field
    initial_project_snapshot: ZedProjectSnapshot | None = None
    cumulative_token_usage: dict[str, int] = Field(default_factory=dict)
    request_token_usage: list[dict[str, int]] | dict[str, dict[str, int]] = Field(
        default_factory=list
    )  # Can be list or dict depending on version
    model: ZedLanguageModel | None = None
    completion_mode: str | None = None
    profile: str | None = None
    exceeded_window_error: Any | None = None
    tool_use_limit_reached: bool = False
