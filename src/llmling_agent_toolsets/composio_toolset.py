"""Composio based toolset implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class ComposioTools(ResourceProvider):
    """Provider for composio tools."""

    def __init__(self, user_id: str, toolsets: list[str], api_key: str | None = None) -> None:
        from composio import Composio
        from composio.core.provider._openai import OpenAIProvider

        super().__init__(name=f"composio_{user_id}")
        self.user_id = user_id
        key = api_key or os.environ.get("COMPOSIO_API_KEY")
        if key:
            self.composio = Composio[OpenAIProvider](api_key=key)
        else:
            self.composio = Composio[OpenAIProvider]()
        self._tools: list[Tool] | None = None
        self._toolkits = toolsets

    def _create_tool_handler(self, tool_slug: str) -> Callable[..., Any]:
        """Create a handler function for a specific tool."""

        def handle_tool_call(**kwargs: Any) -> Any:
            try:
                return self.composio.tools.execute(
                    slug=tool_slug,
                    arguments=kwargs,
                    user_id=self.user_id,
                )
            except Exception:
                logger.exception("Error executing tool", name=tool_slug)
                return {"error": f"Failed to execute tool {tool_slug}"}

        handle_tool_call.__name__ = tool_slug
        return handle_tool_call

    async def get_tools(self) -> list[Tool]:
        """Get tools from composio."""
        # Return cached tools if available
        if self._tools is not None:
            return self._tools

        self._tools = []

        try:
            # Get tools for GitHub toolkit using v3 API
            tools = self.composio.tools.get(self.user_id, toolkits=self._toolkits)

            for tool_def in tools:
                # In v3 SDK, tools are OpenAI formatted by default
                if isinstance(tool_def, dict) and "function" in tool_def:
                    tool_slug = tool_def["function"].get("name", "")
                    if tool_slug:
                        fn = self._create_tool_handler(tool_slug)
                        tool = self.create_tool(fn, schema_override=tool_def["function"])
                        self._tools.append(tool)

        except Exception:
            logger.exception("Error getting Composio tools")
            # Return empty list if there's an error
            self._tools = []

        return self._tools


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        from llmling_agent import Agent

        tools = ComposioTools("user@example.com", toolsets=["github"])
        agent = Agent(model="gpt-5-nano")
        agent.tools.add_provider(tools)
        result = await agent.run("tell me the tools at your disposal")
        print(result)

    asyncio.run(main())
