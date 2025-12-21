"""Data classes for storing agent data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict


if TYPE_CHECKING:
    from datetime import datetime


GroupBy = Literal["agent", "model", "hour", "day"]


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


class MessageData(TypedDict):
    """Formatted message data."""

    role: str
    """Role of the message sender (user/assistant/system)"""

    content: str
    """Content of the message"""

    timestamp: str
    """When the message was sent (ISO format)"""

    model: str | None
    """Name of the model that generated this message"""

    name: str | None
    """Display name of the sender"""

    token_usage: TokenUsage | None
    """Token usage statistics if available"""

    cost: float | None
    """Cost of generating this message in USD"""

    response_time: float | None
    """Time taken to generate the response in seconds"""


class ConversationData(TypedDict):
    """Formatted conversation data."""

    id: str
    """Unique identifier for the conversation"""

    agent: str
    """Name of the agent that handled this conversation"""

    title: str | None
    """AI-generated or user-provided conversation title"""

    start_time: str
    """When the conversation started (ISO format)"""

    messages: list[MessageData]
    """List of messages in this conversation"""

    token_usage: TokenUsage | None
    """Aggregated token usage for the entire conversation"""


@dataclass
class QueryFilters:
    """Filters for conversation queries."""

    agent_name: str | None = None
    """Filter by specific agent name"""

    since: datetime | None = None
    """Only include conversations after this time"""

    query: str | None = None
    """Search term to filter message content"""

    model: str | None = None
    """Filter by model name"""

    limit: int | None = None
    """Maximum number of conversations to return"""


@dataclass
class StatsFilters:
    """Filters for statistics queries."""

    cutoff: datetime
    """Only include data after this time"""

    group_by: GroupBy
    """How to group the statistics (agent/model/hour/day)"""

    agent_name: str | None = None
    """Filter statistics to specific agent"""
