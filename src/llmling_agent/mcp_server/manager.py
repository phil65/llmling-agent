"""MCP server management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self, assert_never, cast

from pydantic_ai import UsageLimits

from llmling_agent.log import get_logger
from llmling_agent.models.content import AudioBase64Content, ImageBase64Content
from llmling_agent.resource_providers import AggregatingResourceProvider, ResourceProvider
from llmling_agent.resource_providers.mcp_provider import MCPResourceProvider
from llmling_agent_config.mcp_server import BaseMCPServerConfig


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from fastmcp.client.elicitation import ElicitResult
    from mcp import types
    from mcp.client.session import RequestContext
    from mcp.types import SamplingMessage

    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.content import BaseContent
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


class MCPManager:
    """Manages MCP server connections and distributes resource providers."""

    def __init__(
        self,
        name: str = "mcp",
        owner: str | None = None,
        servers: Sequence[MCPServerConfig | str] | None = None,
        context: NodeContext | None = None,
        accessible_roots: list[str] | None = None,
    ) -> None:
        self.name = name
        self.owner = owner
        self.servers: list[MCPServerConfig] = []
        for server in servers or []:
            self.add_server_config(server)
        self.context = context
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
            # Setup directly provided servers and context servers concurrently
            tasks = [self._setup_server(server) for server in self.servers]
            if self.context and (cfg := self.context.config) and cfg.mcp_servers:
                tasks.extend(self._setup_server(s) for s in cfg.get_mcp_servers())
            if tasks:
                await asyncio.gather(*tasks)

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
    ) -> None:
        await self.cleanup()

    async def _elicitation_callback(
        self,
        message: str,
        response_type: type[Any],
        params: types.ElicitRequestParams,
        context: RequestContext,
    ) -> ElicitResult[dict[str, Any]] | dict[str, Any] | None:
        """Handle elicitation requests from MCP server."""
        from fastmcp.client.elicitation import ElicitResult
        from mcp.types import ElicitResult as MCPElicitResult, ErrorData

        from llmling_agent.agent.context import AgentContext

        if self.context and isinstance(self.context, AgentContext):
            match await self.context.handle_elicitation(params):
                case MCPElicitResult(action="accept", content=content):
                    return content
                case MCPElicitResult(action="cancel"):
                    return ElicitResult(action="cancel")
                case MCPElicitResult(action="decline"):
                    return ElicitResult(action="decline")
                case MCPElicitResult():
                    msg = "Invalid elicitation result"
                    raise ValueError(msg)
                case ErrorData():
                    return ElicitResult(action="decline")
                case _ as unreachable:
                    assert_never(unreachable)

        return ElicitResult(action="decline")

    async def _sampling_callback(
        self,
        messages: list[SamplingMessage],
        params: types.CreateMessageRequestParams,
        context: RequestContext,
    ) -> str:
        """Handle MCP sampling by creating a new agent with specified preferences."""
        from mcp import types

        from llmling_agent.agent import Agent

        try:
            # Convert messages to prompts for the agent
            prompts: list[BaseContent | str] = []
            for mcp_msg in messages:
                match mcp_msg.content:
                    case types.TextContent(text=text):
                        prompts.append(text)
                    case types.ImageContent(data=data, mimeType=mime_type):
                        our_image = ImageBase64Content(data=data, mime_type=mime_type)
                        prompts.append(our_image)
                    case types.AudioContent(data=data, mimeType=mime_type):
                        fmt = mime_type.removeprefix("audio/")
                        our_audio = AudioBase64Content(data=data, format=fmt)
                        prompts.append(our_audio)

            # Extract model from preferences
            model = None
            if (
                params.modelPreferences
                and params.modelPreferences.hints
                and params.modelPreferences.hints[0].name
            ):
                model = params.modelPreferences.hints[0].name

            # Create usage limits from sampling parameters
            usage_limits = UsageLimits(
                output_tokens_limit=params.maxTokens,
                request_limit=1,  # Single sampling request
            )

            # TODO: Apply temperature from params.temperature
            # Currently no direct way to pass temperature to Agent constructor
            # May need provider-level configuration or runtime model settings

            # Create agent with sampling parameters
            agent = Agent(
                name="mcp-sampling-agent",
                model=model,
                system_prompt=params.systemPrompt or "",
                session=False,  # Don't store history for sampling
            )

            async with agent:
                # Pass all prompts directly to the agent
                result = await agent.run(
                    *prompts,
                    store_history=False,
                    usage_limits=usage_limits,
                )

                return str(result.content)

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
            context=self.context,
            source="pool" if self.owner == "pool" else "node",
            elicitation_callback=self._elicitation_callback,
            sampling_callback=self._sampling_callback,
            accessible_roots=self._accessible_roots,
        )

        # Initialize the provider and add to exit stack
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

        # Create and initialize the provider
        provider = MCPResourceProvider(
            server=config,
            name=f"{self.name}_{config.client_id}",
            owner=self.owner,
            context=self.context,
            source="pool" if self.owner == "pool" else "node",
            elicitation_callback=self._elicitation_callback,
            sampling_callback=self._sampling_callback,
            accessible_roots=self._accessible_roots,
        )

        # Initialize the provider and add to exit stack
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
        args=["run", "/home/phil65/dev/oss/llmling-agent/tests/mcp/server.py"],
    )

    async def main() -> None:
        manager = MCPManager(servers=[cfg])
        async with manager:
            providers = manager.get_mcp_providers()
            print(f"Found {len(providers)} providers")

            if providers:
                provider = providers[0]
                prompts = await provider.get_prompts()
                print(f"Found prompts: {prompts}")

                if prompts:
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
