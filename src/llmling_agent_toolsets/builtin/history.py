"""Provider for history tools."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent.utils.now import get_now


async def search_history(
    ctx: AgentContext,
    query: str | None = None,
    hours: int = 24,
    limit: int = 5,
) -> str:
    """Search conversation history."""
    from llmling_agent_storage.formatters import format_output

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    provider = ctx.storage.get_history_provider()
    results = await provider.get_filtered_conversations(
        query=query,
        period=f"{hours}h",
        limit=limit,
    )
    return format_output(results)


async def show_statistics(
    ctx: AgentContext,
    group_by: Literal["agent", "model", "hour", "day"] = "model",
    hours: int = 24,
) -> str:
    """Show usage statistics for conversations."""
    from llmling_agent_storage.formatters import format_output
    from llmling_agent_storage.models import StatsFilters

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    cutoff = get_now() - timedelta(hours=hours)
    filters = StatsFilters(cutoff=cutoff, group_by=group_by)

    provider = ctx.storage.get_history_provider()
    stats = await provider.get_conversation_stats(filters)

    return format_output(
        {
            "period": f"{hours}h",
            "group_by": group_by,
            "entries": [
                {
                    "name": key,
                    "messages": data["messages"],
                    "total_tokens": data["total_tokens"],
                    "models": sorted(data["models"]),
                }
                for key, data in stats.items()
            ],
        },
        output_format="text",
    )


def create_history_tools() -> list[Tool]:
    """Create tools for history and statistics access."""
    return [
        Tool.from_callable(search_history, source="builtin", category="search"),
        Tool.from_callable(show_statistics, source="builtin", category="read"),
    ]


class HistoryTools(StaticResourceProvider):
    """Provider for history tools."""

    def __init__(self, name: str = "history"):
        super().__init__(name=name, tools=create_history_tools())
