"""Manages message flow between agents/groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage, FormatStyle


@dataclass(frozen=True, kw_only=True)
class MessageStats:
    """Statistics for a single connection."""

    messages: list[ChatMessage[Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def message_count(self) -> int:
        """Number of messages transmitted."""
        return len(self.messages)

    @property
    def last_message_time(self) -> datetime | None:
        """When the last message was sent."""
        return self.messages[-1].timestamp if self.messages else None

    @property
    def token_count(self) -> int:
        """Total tokens used."""
        return sum(
            msg.cost_info.token_usage["total"] for msg in self.messages if msg.cost_info
        )

    @property
    def tool_calls(self) -> list[ToolCallInfo]:
        """Accumulated tool calls going through this connection."""
        result = []
        for msg in self.messages:
            result.extend(msg.tool_calls)
        return result

    @property
    def byte_count(self) -> int:
        """Total bytes transmitted."""
        return sum(len(str(msg.content).encode()) for msg in self.messages)

    def format(
        self,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format the conversation that happened on this connection."""
        return "\n".join(msg.format(style, **kwargs) for msg in self.messages)

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(
            float(msg.cost_info.total_cost) for msg in self.messages if msg.cost_info
        )


@dataclass(frozen=True, kw_only=True)
class TalkStats(MessageStats):
    """Statistics for a single connection."""

    source_name: str | None
    target_names: set[str]


@dataclass(kw_only=True)
class AggregatedMessageStats:
    """Statistics aggregated from multiple connections."""

    stats: Sequence[MessageStats | AggregatedMessageStats] = field(default_factory=list)

    # def __init__(self, stats: list[TalkStats | AggregatedTalkStats]):
    #     self.stats = stats

    @property
    def messages(self) -> list[ChatMessage[Any]]:
        """All messages across all connections, flattened."""
        return [msg for stat in self.stats for msg in stat.messages]

    @property
    def message_count(self) -> int:
        """Total messages across all connections."""
        return len(self.messages)

    @property
    def tool_calls(self) -> list[ToolCallInfo]:
        """Accumulated tool calls going through this connection."""
        result = []
        for msg in self.messages:
            result.extend(msg.tool_calls)
        return result

    @property
    def start_time(self) -> datetime:
        """Total messages across all connections."""
        return self.stats[0].start_time

    @property
    def num_connections(self) -> int:
        """Number of active connections."""
        return len(self.stats)

    @property
    def token_count(self) -> int:
        """Total tokens across all connections."""
        return sum(
            msg.cost_info.token_usage["total"] for msg in self.messages if msg.cost_info
        )

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(
            float(msg.cost_info.total_cost) for msg in self.messages if msg.cost_info
        )

    @property
    def byte_count(self) -> int:
        """Total bytes transmitted."""
        return sum(len(str(msg.content).encode()) for msg in self.messages)

    @property
    def last_message_time(self) -> datetime | None:
        """Most recent message time."""
        if not self.messages:
            return None
        return max(msg.timestamp for msg in self.messages)

    def format(
        self,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format all conversations in the team."""
        return "\n".join(msg.format(style, **kwargs) for msg in self.messages)


@dataclass(kw_only=True)
class AggregatedTalkStats(AggregatedMessageStats):
    """Statistics aggregated from multiple connections."""

    stats: Sequence[TalkStats | AggregatedTalkStats] = field(default_factory=list)

    @cached_property
    def source_names(self) -> set[str]:
        """Set of unique source names recursively."""

        def _collect_source_names(stat: TalkStats | AggregatedTalkStats) -> set[str]:
            """Recursively collect source names."""
            if isinstance(stat, TalkStats):
                return {stat.source_name} if stat.source_name else set()
            # It's a AggregatedTalkStats, recurse
            names = set()
            for s in stat.stats:
                names.update(_collect_source_names(s))
            return names

        names = set()
        for stat in self.stats:
            names.update(_collect_source_names(stat))
        return names

    @cached_property
    def target_names(self) -> set[str]:
        """Set of all target names across connections."""
        return {name for s in self.stats for name in s.target_names}
