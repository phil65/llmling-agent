"""Provider for history tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_history_tools


class HistoryTools(StaticResourceProvider):
    """Provider for history tools."""

    def __init__(self, name: str = "history"):
        super().__init__(name=name, tools=create_history_tools())
