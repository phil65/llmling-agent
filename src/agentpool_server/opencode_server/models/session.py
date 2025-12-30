"""Session related models."""

from __future__ import annotations

from typing import Literal

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeCreatedUpdated  # noqa: TC001


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


class SessionStatus(OpenCodeBaseModel):
    """Status of a session."""

    type: Literal["idle", "busy", "retry"] = "idle"


class Todo(OpenCodeBaseModel):
    """Todo item for a session."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
