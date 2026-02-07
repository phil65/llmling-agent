"""Common/shared models used across multiple domains."""

from typing import Self

from pydantic import Field

from agentpool.utils.time_utils import now_ms
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class TimeCreatedUpdated(OpenCodeBaseModel):
    """Timestamp with created and updated fields (milliseconds)."""

    created: int
    updated: int


class TimeCreated(OpenCodeBaseModel):
    """Timestamp with created field only (milliseconds)."""

    created: int

    @classmethod
    def now(cls) -> Self:
        return cls(created=now_ms())


class TimeStartEnd(OpenCodeBaseModel):
    """Timestamp with start and optional end (milliseconds)."""

    start: int
    end: int | None = None


class ModelRef(OpenCodeBaseModel):
    """Reference to a provider model (provider_id + model_id)."""

    provider_id: str
    model_id: str


class TokenCache(OpenCodeBaseModel):
    """Token cache information."""

    read: int = 0
    write: int = 0


class Tokens(OpenCodeBaseModel):
    """Token usage information."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache: TokenCache = Field(default_factory=TokenCache)


class FileDiff(OpenCodeBaseModel):
    """A file diff entry."""

    file: str
    before: str
    after: str
    additions: int
    deletions: int
