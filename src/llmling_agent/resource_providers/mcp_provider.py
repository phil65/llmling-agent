"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent_config.mcp_server import BaseMCPServerConfig
from llmling_agent_config.resources import ResourceInfo


if TYPE_CHECKING:
    from typing import Literal

    from fastmcp.client.sampling import SamplingHandler

    from llmling_agent.prompts.prompts import MCPClientPrompt
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
        source: Literal["pool", "node"] = "node",
        sampling_callback: SamplingHandler[Any, Any] | None = None,
        accessible_roots: list[str] | None = None,
    ) -> None:
        from llmling_agent.mcp_server import MCPClient

        super().__init__(name, owner=owner)
        self.server = BaseMCPServerConfig.from_string(server) if isinstance(server, str) else server
        self.source = source
        self.exit_stack = AsyncExitStack()
        self._accessible_roots = accessible_roots
        self._sampling_callback = sampling_callback

        # Tool caching
        self._tools_cache: list[Tool] | None = None
        self._saved_enabled_states: dict[str, bool] = {}

        # Prompt caching
        self._prompts_cache: list[MCPClientPrompt] | None = None

        # Resource caching
        self._resources_cache: list[ResourceInfo] | None = None

        self.client = MCPClient(
            config=self.server,
            sampling_callback=self._sampling_callback,
            accessible_roots=self._accessible_roots,
            tool_change_callback=self._on_tools_changed,
            prompt_change_callback=self._on_prompts_changed,
            resource_change_callback=self._on_resources_changed,
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

    async def __aexit__(self, *args: object) -> None:
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
        self._saved_enabled_states = {t.name: t.enabled for t in self._tools_cache or []}
        self._tools_cache = None

    async def _on_prompts_changed(self) -> None:
        """Callback when prompts change on the MCP server."""
        logger.info("MCP prompt list changed, refreshing provider cache")
        self._prompts_cache = None

    async def _on_resources_changed(self) -> None:
        """Callback when resources change on the MCP server."""
        logger.info("MCP resource list changed, refreshing provider cache")
        self._resources_cache = None

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

            # Restore enabled states from saved states
            for tool_info in all_tools:
                if tool_info.name in self._saved_enabled_states:
                    tool_info.enabled = self._saved_enabled_states[tool_info.name]

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

    async def refresh_prompts_cache(self) -> None:
        """Refresh the prompts cache by fetching from client."""
        from llmling_agent.prompts.prompts import MCPClientPrompt

        try:
            result = await self.client.list_prompts()
            all_prompts: list[MCPClientPrompt] = []

            for prompt in result:
                try:
                    converted = MCPClientPrompt.from_fastmcp(self.client, prompt)
                    all_prompts.append(converted)
                except Exception:
                    logger.exception("Failed to convert prompt", name=prompt.name)
                    continue

            self._prompts_cache = all_prompts
            logger.debug("Refreshed MCP prompts cache", num_prompts=len(all_prompts))
        except Exception:
            logger.exception("Failed to refresh MCP prompts cache")
            self._prompts_cache = []

    async def get_prompts(self) -> list[MCPClientPrompt]:  # type: ignore
        """Get cached prompts, refreshing if necessary."""
        if self._prompts_cache is None:
            await self.refresh_prompts_cache()

        return self._prompts_cache or []

    async def refresh_resources_cache(self) -> None:
        """Refresh the resources cache by fetching from client."""
        try:
            result = await self.client.list_resources()
            all_resources: list[ResourceInfo] = []

            for resource in result:
                try:
                    converted = await ResourceInfo.from_mcp_resource(resource)
                    all_resources.append(converted)
                except Exception:
                    logger.exception("Failed to convert resource", name=resource.name)
                    continue

            self._resources_cache = all_resources
            logger.debug("Refreshed MCP resources cache", num_resources=len(all_resources))
        except Exception:
            logger.exception("Failed to refresh MCP resources cache")
            self._resources_cache = []

    async def get_resources(self) -> list[ResourceInfo]:
        """Get cached resources, refreshing if necessary."""
        if self._resources_cache is None:
            await self.refresh_resources_cache()

        return self._resources_cache or []


if __name__ == "__main__":
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    cfg = StdioMCPServerConfig(
        command="uv",
        args=["run", "/home/phil65/dev/oss/llmling-agent/tests/mcp_server/server.py"],
    )

    async def main() -> None:
        manager = MCPResourceProvider(cfg)
        async with manager:
            prompts = await manager.get_prompts()
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
