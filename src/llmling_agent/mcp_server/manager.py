"""MCP server management for LLMling agents."""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic_ai import BinaryContent, BinaryImage, UsageLimits

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import AggregatingResourceProvider, ResourceProvider
from llmling_agent.resource_providers.mcp_provider import MCPResourceProvider
from llmling_agent_config.mcp_server import BaseMCPServerConfig


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from mcp import types
    from mcp.shared.context import RequestContext
    from mcp.types import SamplingMessage
    from pydantic_ai import UserContent

    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


class MCPManager:
    """Manages MCP server connections and distributes resource providers."""

    def __init__(
        self,
        name: str = "mcp",
        owner: str | None = None,
        servers: Sequence[MCPServerConfig | str] | None = None,
        accessible_roots: list[str] | None = None,
    ) -> None:
        self.name = name
        self.owner = owner
        self.servers: list[MCPServerConfig] = []
        for server in servers or []:
            self.add_server_config(server)
        self.providers: list[MCPResourceProvider] = []
        self.aggregating_provider = AggregatingResourceProvider(
            providers=cast(list[ResourceProvider], self.providers),
            name=f"{name}_aggregated",
        )
        self.exit_stack = AsyncExitStack()
        self._accessible_roots = accessible_roots

    def add_server_config(self, cfg: MCPServerConfig | str) -> None:
        """Add a new MCP server to the manager."""
        resolved = BaseMCPServerConfig.from_string(cfg) if isinstance(cfg, str) else cfg
        self.servers.append(resolved)

    def __repr__(self) -> str:
        return f"MCPManager(name={self.name!r}, servers={len(self.servers)})"

    async def __aenter__(self) -> Self:
        try:
            if tasks := [self._setup_server(server) for server in self.servers]:
                await asyncio.gather(*tasks)
        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize MCP manager"
            raise RuntimeError(msg) from e

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.cleanup()

    async def _sampling_callback(
        self,
        messages: list[SamplingMessage],
        params: types.CreateMessageRequestParams,
        context: RequestContext[Any, Any, Any],
    ) -> str:
        """Handle MCP sampling by creating a new agent with specified preferences."""
        from mcp import types

        from llmling_agent.agents import Agent

        # Convert messages to prompts for the agent
        prompts: list[UserContent] = []
        for mcp_msg in messages:
            match mcp_msg.content:
                case types.TextContent(text=text):
                    prompts.append(text)
                case types.ImageContent(data=data, mimeType=mime_type):
                    binary_data = base64.b64decode(data)
                    prompts.append(BinaryImage(data=binary_data, media_type=mime_type))
                case types.AudioContent(data=data, mimeType=mime_type):
                    binary_data = base64.b64decode(data)
                    prompts.append(BinaryContent(data=binary_data, media_type=mime_type))

        # Extract model from preferences
        model = None
        if (prefs := params.modelPreferences) and prefs.hints and prefs.hints[0].name:
            model = prefs.hints[0].name
        # Create usage limits from sampling parameters
        limits = UsageLimits(output_tokens_limit=params.maxTokens, request_limit=1)
        # TODO: Apply temperature from params.temperature
        sys_prompt = params.systemPrompt or ""
        agent = Agent(name="sampling-agent", model=model, system_prompt=sys_prompt, session=False)
        try:
            async with agent:
                result = await agent.run(*prompts, store_history=False, usage_limits=limits)
                return result.content

        except Exception as e:
            logger.exception("Sampling failed")
            return f"Sampling failed: {e!s}"

    async def _setup_server(self, config: MCPServerConfig) -> None:
        """Set up a single MCP server resource provider."""
        if not config.enabled:
            return

        provider = MCPResourceProvider(
            server=config,
            name=f"{self.name}_{config.client_id}",
            owner=self.owner,
            source="pool" if self.owner == "pool" else "node",
            sampling_callback=self._sampling_callback,
            accessible_roots=self._accessible_roots,
        )
        provider = await self.exit_stack.enter_async_context(provider)
        self.providers.append(provider)

    def get_mcp_providers(self) -> list[MCPResourceProvider]:
        """Get all MCP resource providers managed by this manager."""
        return list(self.providers)

    def get_aggregating_provider(self) -> AggregatingResourceProvider:
        """Get the aggregating provider that contains all MCP providers."""
        return self.aggregating_provider

    async def setup_server_runtime(self, config: MCPServerConfig) -> MCPResourceProvider:
        """Set up a single MCP server at runtime while manager is running.

        Returns:
            The newly created and initialized MCPResourceProvider
        """
        if not config.enabled:
            msg = f"Server config {config.client_id} is disabled"
            raise ValueError(msg)

        # Add the config first
        self.add_server_config(config)
        provider = MCPResourceProvider(
            server=config,
            name=f"{self.name}_{config.client_id}",
            owner=self.owner,
            source="pool" if self.owner == "pool" else "node",
            sampling_callback=self._sampling_callback,
            accessible_roots=self._accessible_roots,
        )
        provider = await self.exit_stack.enter_async_context(provider)
        self.providers.append(provider)
        # Note: AggregatingResourceProvider automatically sees the new provider
        # since it references self.providers list

        return provider

    async def cleanup(self) -> None:
        """Clean up all MCP connections and providers."""
        try:
            try:
                # Clean up exit stack (which includes MCP providers)
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

            self.providers.clear()

        except Exception as e:
            msg = "Error during MCP manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    @property
    def active_servers(self) -> list[str]:
        """Get IDs of active servers."""
        return [provider.server.client_id for provider in self.providers]


if __name__ == "__main__":
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    cfg = StdioMCPServerConfig(
        command="uv",
        args=["run", "/home/phil65/dev/oss/llmling-agent/tests/mcp_server/server.py"],
    )

    async def main() -> None:
        manager = MCPManager(servers=[cfg])
        async with manager:
            providers = manager.get_mcp_providers()
            print(f"Found {len(providers)} providers")
            provider = providers[0]
            prompts = await provider.get_prompts()
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

    asyncio.run(main())
