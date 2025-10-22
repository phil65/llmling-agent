"""Provider for history tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


def create_history_tools() -> list[Tool]:
    """Create tools for history and statistics access."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.search_history,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.show_statistics,
            source="builtin",
            category="read",
        ),
    ]


class HistoryTools(StaticResourceProvider):
    """Provider for history tools."""

    def __init__(self, name: str = "history"):
        super().__init__(name=name, tools=create_history_tools())
