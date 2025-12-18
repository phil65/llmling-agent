"""McpRun based toolset implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from schemez import OpenAIFunctionDefinition

from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from llmling_agent.tools.base import Tool


class McpRunTools(ResourceProvider):
    """Provider for MCP.run tools."""

    def __init__(self, entity_id: str, session_id: str | None = None) -> None:
        from mcp_run import Client, ClientConfig  # type: ignore[import-untyped]

        super().__init__(name=entity_id)
        id_ = session_id or os.environ.get("MCP_RUN_SESSION_ID")
        config = ClientConfig()
        self.client = Client(session_id=id_, config=config)
        self._tools: list[Tool] | None = None

    async def get_tools(self) -> list[Tool]:
        """Get tools from MCP.run."""
        # Return cached tools if available
        if self._tools is not None:
            return self._tools

        self._tools = []
        for name, tool in self.client.tools.items():

            async def run(tool_name: str = name, **input_dict: Any) -> CallToolResult:
                async with self.client.mcp_sse().connect() as session:
                    return await session.call_tool(tool_name, arguments=input_dict)  # type: ignore[no-any-return]

            run.__name__ = name
            wrapped_tool = self.create_tool(
                run, schema_override=cast(OpenAIFunctionDefinition, tool.input_schema)
            )
            self._tools.append(wrapped_tool)

        return self._tools


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        tools = McpRunTools("default")
        fns = await tools.get_tools()
        print(fns)

    asyncio.run(main())
