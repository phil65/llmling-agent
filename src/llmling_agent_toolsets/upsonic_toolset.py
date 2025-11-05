"""Upsonic based toolset implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from llmling.core.log import get_logger

from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from pydantic import SecretStr

    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class UpsonicTools(ResourceProvider):
    """Provider for upsonic tools."""

    def __init__(self, api_key: SecretStr | None = None, base_url: str | None = None):
        from upsonic import Tiger, Tiger_Admin

        super().__init__(name="tiger")
        key = api_key.get_secret_value() if api_key else os.getenv("UPSONIC_API_KEY")
        if base_url:
            assert key
            self.tiger_instance = Tiger_Admin(api_url=base_url, access_key=key)
        else:
            self.tiger_instance = Tiger()
        self._tools: list[Tool] | None = None

    async def get_tools(self) -> list[Tool]:
        """Get tools from entry points."""
        # Return cached tools if available
        if self._tools is not None:
            return self._tools

        self._tools = []
        return self._tools


if __name__ == "__main__":
    import asyncio

    async def main():
        from llmling_agent import Agent

        tools = UpsonicTools()
        agent = Agent(model="gpt-5-nano")
        agent.tools.add_provider(tools)
        result = await agent.run("tell me the tools at your disposal")
        print(result)

    asyncio.run(main())
