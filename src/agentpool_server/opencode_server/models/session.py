"""Session related models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import TimeCreatedUpdated  # noqa: TC001


class SessionRevert(OpenCodeBaseModel):
    """Revert information for a session."""

    message_id: str = Field(alias="messageID")
    diff: str | None = None
    part_id: str | None = Field(default=None, alias="partID")
    snapshot: str | None = None


class SessionShare(OpenCodeBaseModel):
    """Share information for a session."""

    url: str


class Session(OpenCodeBaseModel):
    """Session information."""

    id: str
    project_id: str = Field(alias="projectID")
    directory: str
    title: str
    version: str = "1"
    time: TimeCreatedUpdated
    parent_id: str | None = Field(default=None, alias="parentID")
    revert: SessionRevert | None = None
    share: SessionShare | None = None


class SessionCreateRequest(OpenCodeBaseModel):
    """Request body for creating a session."""

    parent_id: str | None = Field(default=None, alias="parentID")
    title: str | None = None


class SessionUpdateRequest(OpenCodeBaseModel):
    """Request body for updating a session."""

    title: str | None = None


class SessionStatus(OpenCodeBaseModel):
    """Status of a session."""

    type: Literal["idle", "busy", "retry"] = "idle"


class Todo(OpenCodeBaseModel):
    """Todo item for a session."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
