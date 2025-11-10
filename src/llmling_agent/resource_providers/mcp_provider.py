"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Self

from llmling_agent.log import get_logger
from llmling_agent.mcp_server import MCPClient
from llmling_agent.prompts.prompts import Prompt
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent_config.mcp_server import BaseMCPServerConfig
from llmling_agent_config.resources import ResourceInfo


if TYPE_CHECKING:
    from types import TracebackType
    from typing import Literal

    from fastmcp.client.elicitation import ElicitationHandler
    from fastmcp.client.sampling import ClientSamplingHandler

    from llmling_agent.mcp_server.client import ContextualProgressHandler
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


class MCPResourceProvider(ResourceProvider):
    """Resource provider for a single MCP server."""

    def __init__(
        self,
        server: MCPServerConfig | str,
        name: str = "mcp",
        owner: str | None = None,
        context: NodeContext | None = None,
        source: Literal["pool", "node"] = "node",
        elicitation_callback: ElicitationHandler | None = None,
        sampling_callback: ClientSamplingHandler | None = None,
        progress_handler: ContextualProgressHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        super().__init__(name, owner=owner)
        self.server = (
            BaseMCPServerConfig.from_string(server) if isinstance(server, str) else server
        )
        self.context = context
        self.source = source
        self.exit_stack = AsyncExitStack()
        self._progress_handler = progress_handler
        self._accessible_roots = accessible_roots
        self._elicitation_callback = elicitation_callback
        self._sampling_callback = sampling_callback

        # Tool caching
        self._tools_cache: list[Tool] | None = None

        self.client = MCPClient(
            config=self.server,
            elicitation_callback=self._elicitation_callback,
            sampling_callback=self._sampling_callback,
            progress_handler=self._progress_handler,
            accessible_roots=self._accessible_roots,
            tool_change_callback=self._on_tools_changed,
        )

    def __repr__(self) -> str:
        return f"MCPResourceProvider({self.server!r}, source={self.source!r})"

    async def __aenter__(self) -> Self:
        try:
            await self.exit_stack.enter_async_context(self.client)
        except Exception as e:
            # Clean up in case of error
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize MCP manager"
            raise RuntimeError(msg) from e

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        try:
            try:
                # Clean up exit stack (which includes MCP clients)
                await self.exit_stack.aclose()
            except RuntimeError as e:
                if "different task" in str(e):
                    # Handle task context mismatch
                    current_task = asyncio.current_task()
                    if current_task:
                        loop = asyncio.get_running_loop()
                        await loop.create_task(self.exit_stack.aclose())
                else:
                    raise

        except Exception as e:
            msg = "Error during MCP manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    async def _on_tools_changed(self) -> None:
        """Callback when tools change on the MCP server."""
        logger.info("MCP tool list changed, refreshing provider cache")
        self._tools_cache = None  # Invalidate cache

    async def refresh_tools_cache(self) -> None:
        """Refresh the tools cache by fetching from client."""
        try:
            # Get fresh tools from client
            mcp_tools = await self.client.list_tools()
            all_tools: list[Tool] = []

            for tool in mcp_tools:
                try:
                    tool_info = self.client.convert_tool(tool)
                    all_tools.append(tool_info)
                except Exception:
                    logger.exception("Failed to create MCP tool", name=tool.name)
                    continue

            self._tools_cache = all_tools
            logger.debug("Refreshed MCP tools cache", num_tools=len(all_tools))
        except Exception:
            logger.exception("Failed to refresh MCP tools cache")
            self._tools_cache = []

    async def get_tools(self) -> list[Tool]:
        """Get cached tools, refreshing if necessary."""
        if self._tools_cache is None:
            await self.refresh_tools_cache()

        return self._tools_cache or []

    async def list_prompts(self) -> list[Prompt]:
        """Get all available prompts from MCP servers."""
        try:
            result = await self.client.list_prompts()
            client_prompts: list[Prompt] = []
            for prompt in result:
                try:
                    converted = Prompt.from_fastmcp(self.client, prompt)
                    client_prompts.append(converted)
                except Exception:
                    logger.exception("Failed to convert prompt", name=prompt.name)
        except Exception:
            logger.exception("Failed to get prompts from MCP server")
            return []
        else:
            return client_prompts

    async def list_resources(self) -> list[ResourceInfo]:
        """Get all available resources from MCP servers."""
        try:
            result = await self.client.list_resources()
            client_resources: list[ResourceInfo] = []
            for resource in result:
                try:
                    converted = await ResourceInfo.from_mcp_resource(resource)
                    client_resources.append(converted)
                except Exception:
                    logger.exception("Failed to convert resource", name=resource.name)
        except Exception:
            logger.exception("Failed to get resources from MCP server")
            return []
        else:
            return client_resources


if __name__ == "__main__":
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    cfg = StdioMCPServerConfig(
        command="uv",
        args=["run", "/home/phil65/dev/oss/llmling-agent/tests/mcp/server.py"],
    )

    async def main():
        manager = MCPResourceProvider(cfg)
        async with manager:
            prompts = await manager.list_prompts()
            print(f"Found prompts: {prompts}")

            # Test static prompt (no arguments)
            static_prompt = next(p for p in prompts if p.name == "static_prompt")
            print(f"\n--- Testing static prompt: {static_prompt} ---")
            components = await static_prompt.get_components()
            assert components, "No prompt components found"
            print(f"Found {len(components)} prompt components:")
            for i, component in enumerate(components):
                comp_type = type(component).__name__
                print(f"  {i + 1}. {comp_type}: {component.content}")

            # Test dynamic prompt (with arguments)
            dynamic_prompt = next(p for p in prompts if p.name == "dynamic_prompt")
            print(f"\n--- Testing dynamic prompt: {dynamic_prompt} ---")
            components = await dynamic_prompt.get_components(
                arguments={"some_arg": "Hello, world!"}
            )
            assert components, "No prompt components found"
            print(f"Found {len(components)} prompt components:")
            for i, component in enumerate(components):
                comp_type = type(component).__name__
                print(f"  {i + 1}. {comp_type}: {component.content}")

    asyncio.run(main())
