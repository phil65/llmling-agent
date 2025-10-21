"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self

from llmling.prompts import PromptMessage, StaticPrompt
from mcp.types import TextResourceContents

from llmling_agent.log import get_logger
from llmling_agent.mcp_server.client import MCPClient
from llmling_agent.models.content import AudioBase64Content, ImageBase64Content
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)
from llmling_agent_config.resources import ResourceInfo
from llmling_agent_providers.base import UsageLimits


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastmcp.client.elicitation import ElicitResult
    from mcp import types
    from mcp.client.session import RequestContext
    from mcp.types import Prompt as MCPPrompt, Resource as MCPResource, SamplingMessage

    from llmling_agent.mcp_server.client import ContextualProgressHandler
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.content import BaseContent
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


async def convert_mcp_prompt(client: MCPClient, prompt: MCPPrompt) -> StaticPrompt:
    """Convert MCP prompt to StaticPrompt."""
    from mcp.types import TextContent

    result = await client.get_prompt(prompt.name)
    messages = [
        PromptMessage(role="system", content=message.content.text)
        for message in result.messages
        if isinstance(message.content, TextContent | TextResourceContents)
    ]
    desc = prompt.description or "No description provided"
    return StaticPrompt(name=prompt.name, description=desc, messages=messages)


async def convert_mcp_resource(resource: MCPResource) -> ResourceInfo:
    """Convert MCP resource to ResourceInfo."""
    return ResourceInfo(
        name=resource.name, uri=str(resource.uri), description=resource.description
    )


class MCPManager(ResourceProvider):
    """Manages MCP server connections and tools."""

    def __init__(
        self,
        name: str = "mcp",
        owner: str | None = None,
        servers: Sequence[MCPServerConfig | str] | None = None,
        context: NodeContext | None = None,
        progress_handler: ContextualProgressHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        super().__init__(name, owner=owner)
        self.servers: list[MCPServerConfig] = []
        for server in servers or []:
            self.add_server_config(server)
        self.context = context
        self.clients: dict[str, MCPClient] = {}
        self.exit_stack = AsyncExitStack()
        self._progress_handler = progress_handler
        self._accessible_roots = accessible_roots

    def add_server_config(self, server: MCPServerConfig | str):
        """Add a new MCP server to the manager."""
        match server:
            case str() if server.startswith("http") and server.endswith("/sse"):
                resolved: MCPServerConfig = SSEMCPServerConfig(url=server)
            case str() if server.startswith("http"):
                resolved = StreamableHTTPMCPServerConfig(url=server)
            case str():
                resolved = StdioMCPServerConfig.from_string(server)
            case _:
                resolved = server
        self.servers.append(resolved)

    def __repr__(self) -> str:
        return f"MCPManager({self.servers!r})"

    async def __aenter__(self) -> Self:
        try:
            # Setup directly provided servers and context servers concurrently
            tasks = [self.setup_server(server) for server in self.servers]
            if self.context and self.context.config and self.context.config.mcp_servers:
                tasks.extend(
                    self.setup_server(server)
                    for server in self.context.config.get_mcp_servers()
                )

            if tasks:
                await asyncio.gather(*tasks)

        except Exception as e:
            # Clean up in case of error
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize MCP manager"
            raise RuntimeError(msg) from e

        return self

    async def __aexit__(self, *exc):
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
        from mcp import types

        from llmling_agent.agent.context import AgentContext

        if self.context and isinstance(self.context, AgentContext):
            legacy_result = await self.context.handle_elicitation(params)
            # Convert legacy MCP result to FastMCP format
            if isinstance(legacy_result, types.ElicitResult):
                if legacy_result.action == "accept" and legacy_result.content:
                    return legacy_result.content
                return ElicitResult(action=legacy_result.action)
            if isinstance(legacy_result, types.ErrorData):
                return ElicitResult(action="cancel")
            return ElicitResult(action="decline")

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

    async def setup_server(self, config: MCPServerConfig) -> None:
        """Set up a single MCP server connection."""
        if not config.enabled:
            return

        client = MCPClient(
            elicitation_callback=self._elicitation_callback,
            sampling_callback=self._sampling_callback,
            progress_handler=self._progress_handler,
            accessible_roots=self._accessible_roots,
        )
        client = await self.exit_stack.enter_async_context(client)
        await client.connect(config)

        self.clients[config.client_id] = client

    async def get_tools(self) -> list[Tool]:
        """Get all tools from all connected servers."""
        all_tools: list[Tool] = []

        for client in self.clients.values():
            for tool in client._available_tools:
                try:
                    tool_info = client.convert_tool(tool)
                    all_tools.append(tool_info)
                    logger.debug("Registered MCP tool: %s", tool.name)
                except Exception:
                    msg = "Failed to create tool from MCP tool: %s"
                    logger.exception(msg, tool.name)
                    continue

        return all_tools

    async def list_prompts(self) -> list[StaticPrompt]:
        """Get all available prompts from MCP servers."""

        async def get_client_prompts(client: MCPClient) -> list[StaticPrompt]:
            try:
                result = await client.list_prompts()
                client_prompts: list[StaticPrompt] = []
                for prompt in result:
                    try:
                        converted = await convert_mcp_prompt(client, prompt)
                        client_prompts.append(converted)
                    except Exception:
                        logger.exception("Failed to convert prompt: %s", prompt.name)
            except Exception:
                logger.exception("Failed to get prompts from MCP server")
                return []
            else:
                return client_prompts

        results = await asyncio.gather(*[
            get_client_prompts(client) for client in self.clients.values()
        ])
        return [prompt for client_prompts in results for prompt in client_prompts]

    async def list_resources(self) -> list[ResourceInfo]:
        """Get all available resources from MCP servers."""

        async def get_client_resources(client: MCPClient) -> list[ResourceInfo]:
            try:
                result = await client.list_resources()
                client_resources: list[ResourceInfo] = []
                for resource in result:
                    try:
                        converted = await convert_mcp_resource(resource)
                        client_resources.append(converted)
                    except Exception:
                        logger.exception("Failed to convert resource: %s", resource.name)
            except Exception:
                logger.exception("Failed to get resources from MCP server")
                return []
            else:
                return client_resources

        results = await asyncio.gather(
            *[get_client_resources(client) for client in self.clients.values()],
            return_exceptions=False,
        )
        return [resource for client_resources in results for resource in client_resources]

    async def cleanup(self) -> None:
        """Clean up all MCP connections."""
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

            self.clients.clear()

        except Exception as e:
            msg = "Error during MCP manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    @property
    def active_servers(self) -> list[str]:
        """Get IDs of active servers."""
        return list(self.clients)
