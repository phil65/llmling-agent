"""Provider for history tools."""

from __future__ import annotations

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import StaticResourceProvider


async def search_history(
    ctx: AgentContext,
    query: str | None = None,
    hours: int = 24,
    limit: int = 5,
) -> str:
    """Search conversation history."""
    from llmling_agent_storage.formatters import format_output

    if not ctx.pool:
        return "No agent pool available for history search"
    provider = ctx.pool.storage.get_history_provider()
    results = await provider.get_filtered_conversations(
        query=query,
        period=f"{hours}h",
        limit=limit,
    )
    return format_output(results)


class HistoryTools(StaticResourceProvider):
    """Provider for history tools."""

    def __init__(self, name: str = "history") -> None:
        super().__init__(name=name)
        self._tools = [
            self.create_tool(search_history, category="search", read_only=True, idempotent=True),
        ]
