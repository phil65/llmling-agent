"""Search toolset implementation using searchly providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from searchly.base import NewsSearchProvider, WebSearchProvider

    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class SearchTools(ResourceProvider):
    """Provider for web and news search tools."""

    def __init__(
        self,
        web_search: WebSearchProvider | None = None,
        news_search: NewsSearchProvider | None = None,
    ) -> None:
        """Initialize search tools provider.

        Args:
            web_search: Web search provider instance.
            news_search: News search provider instance.
        """
        super().__init__(name="search")
        self._web_provider = web_search
        self._news_provider = news_search

    async def get_tools(self) -> list[Tool]:
        """Get search tools from configured providers."""
        tools: list[Tool] = []
        if self._web_provider:
            tools.append(
                self.create_tool(
                    self._web_provider.web_search,
                    read_only=True,
                    idempotent=True,
                    open_world=True,
                )
            )
        if self._news_provider:
            tools.append(
                self.create_tool(
                    self._news_provider.news_search,
                    read_only=True,
                    idempotent=True,
                    open_world=True,
                )
            )
        return tools
