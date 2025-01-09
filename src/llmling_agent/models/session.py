from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from llmling_agent.common_types import MessageRole  # noqa: TC001
from llmling_agent.history.filters import parse_time_period


class SessionQuery(BaseModel):
    """Query configuration for session recovery."""

    name: str | None = None
    """Session identifier to match."""

    agents: set[str] | None = None
    """Filter by agent names."""

    since: str | None = None
    """Time period to look back (e.g. "1h", "2d")."""

    until: str | None = None
    """Time period to look up to."""

    contains: str | None = None
    """Filter by message content."""

    roles: set[MessageRole] | None = None
    """Only include specific message roles."""

    limit: int | None = None
    """Maximum number of messages to return."""

    include_forwarded: bool = True
    """Whether to include messages forwarded through agents."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def get_time_cutoff(self) -> datetime | None:
        """Get datetime from time period string."""
        if not self.since:
            return None
        delta = parse_time_period(self.since)
        return datetime.now() - delta
