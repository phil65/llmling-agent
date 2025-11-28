"""Memory configuration for agent memory and history handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field
from schemez import Schema


MessageRole = Literal["user", "assistant"]


if TYPE_CHECKING:
    from datetime import datetime


class MemoryConfig(Schema):
    """Configuration for agent memory and history handling."""

    enable: bool = Field(default=True, title="Enable memory")
    """Whether to enable history tracking."""

    max_tokens: int | None = Field(
        default=None,
        examples=[1000, 4000, 8000],
        title="Maximum context tokens",
    )
    """Maximum number of tokens to keep in context window."""

    max_messages: int | None = Field(
        default=None,
        examples=[10, 50, 100],
        title="Maximum message count",
    )
    """Maximum number of messages to keep in context window."""

    session: SessionQuery | None = Field(default=None, title="Session query")
    """Query configuration for loading previous session."""

    provider: str | None = Field(
        default=None,
        examples=["sqlite", "memory", "file"],
        title="Storage provider",
    )
    """Override default storage provider for this agent.
    If None, uses manifest's default provider or first available."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_value(cls, value: bool | int | str | SessionQuery | UUID | None) -> Self:
        """Create MemoryConfig from any value."""
        match value:
            case False:
                return cls(max_messages=0)
            case int():
                return cls(max_tokens=value)
            case str() | UUID():
                return cls(session=SessionQuery(name=str(value)))
            case SessionQuery():
                return cls(session=value)
            case None | True:
                return cls()
            case _:
                raise ValueError(f"Invalid memory configuration: type: {value}")


class SessionQuery(Schema):
    """Query configuration for session recovery."""

    name: str | None = Field(
        default=None,
        examples=["main_session", "user_123", "conversation_01"],
        title="Session name",
    )
    """Session identifier to match."""

    agents: set[str] | None = Field(default=None, title="Agent filter")
    """Filter by agent names."""

    since: str | None = Field(
        default=None,
        examples=["1h", "2d", "1w"],
        title="Time period lookback",
    )
    """Time period to look back (e.g. "1h", "2d")."""

    until: str | None = Field(default=None, examples=["1h", "2d", "1w"], title="Time period limit")
    """Time period to look up to."""

    contains: str | None = Field(
        default=None,
        examples=["error", "important", "task completed"],
        title="Content filter",
    )
    """Filter by message content."""

    roles: set[MessageRole] | None = Field(default=None, title="Role filter")
    """Only include specific message roles."""

    limit: int | None = Field(default=None, examples=[10, 50, 100], title="Message limit")
    """Maximum number of messages to return."""

    include_forwarded: bool = Field(default=True, title="Include forwarded messages")
    """Whether to include messages forwarded through agents."""

    model_config = ConfigDict(frozen=True)

    def get_time_cutoff(self) -> datetime | None:
        """Get datetime from time period string."""
        from llmling_agent.utils.now import get_now
        from llmling_agent.utils.parse_time import parse_time_period

        if not self.since:
            return None
        delta = parse_time_period(self.since)
        return get_now() - delta
