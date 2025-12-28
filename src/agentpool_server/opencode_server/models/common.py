"""Common/shared models used across multiple domains."""

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class TimeCreatedUpdated(OpenCodeBaseModel):
    """Timestamp with created and updated fields."""

    created: float
    updated: float


class TimeCreated(OpenCodeBaseModel):
    """Timestamp with created field only."""

    created: float


class TimeStartEnd(OpenCodeBaseModel):
    """Timestamp with start and optional end."""

    start: float
    end: float | None = None
