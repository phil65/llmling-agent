"""Session related models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeCreatedUpdated  # noqa: TC001


class SessionSummary(OpenCodeBaseModel):
    """Summary information for a session."""

    additions: int | None = None
    deletions: int | None = None
    files: int | None = None
    diffs: list[Any] = Field(default_factory=list)


class SessionRevert(OpenCodeBaseModel):
    """Revert information for a session."""

    message_id: str
    diff: str | None = None
    part_id: str | None = None
    snapshot: str | None = None


class SessionShare(OpenCodeBaseModel):
    """Share information for a session."""

    url: str


class Session(OpenCodeBaseModel):
    """Session information."""

    id: str
    project_id: str
    directory: str
    title: str
    version: str = "1"
    time: TimeCreatedUpdated
    parent_id: str | None = None
    summary: SessionSummary | None = None
    revert: SessionRevert | None = None
    share: SessionShare | None = None


class SessionCreateRequest(OpenCodeBaseModel):
    """Request body for creating a session."""

    parent_id: str | None = None
    title: str | None = None


class SessionUpdateRequest(OpenCodeBaseModel):
    """Request body for updating a session."""

    title: str | None = None


class SessionForkRequest(OpenCodeBaseModel):
    """Request body for forking a session."""

    message_id: str | None = None
    """Optional message ID to fork from. If provided, only messages up to and including
    this message will be copied to the forked session. If None, all messages are copied."""


class SessionInitRequest(OpenCodeBaseModel):
    """Request body for initializing a session (creating AGENTS.md)."""

    model_id: str | None = None
    """Optional model ID to use for the init task."""

    provider_id: str | None = None
    """Optional provider ID to use for the init task."""


class SummarizeRequest(OpenCodeBaseModel):
    """Request body for summarizing a session.

    Matches OpenCode's compaction API. If model info is provided, uses that model
    for the summary generation. If 'auto' is True, automatically selects the model.
    """

    model_id: str | None = None
    """Optional model ID to use for summary generation."""

    provider_id: str | None = None
    """Optional provider ID to use for summary generation."""

    auto: bool | None = None
    """If True, automatically select the model for summarization."""


class SessionStatus(OpenCodeBaseModel):
    """Status of a session."""

    type: Literal["idle", "busy", "retry"] = "idle"


class Todo(OpenCodeBaseModel):
    """Todo item for a session."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
