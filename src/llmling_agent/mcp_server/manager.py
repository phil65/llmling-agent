"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Self

from llmling import LLMCallableTool
from llmling.prompts import PromptMessage, StaticPrompt

from llmling_agent.log import get_logger
from llmling_agent.mcp_server.client import MCPClient
from llmling_agent.models.mcp_server import MCPServerConfig, SSEMCPServer, StdioMCPServer
from llmling_agent.models.resources import ResourceInfo
from llmling_agent.resource_providers.base import ResourceProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import Prompt as MCPPrompt, Resource as MCPResource

    from llmling_agent.messaging.messagenode import NodeContext
    from llmling_agent.tools.base import ToolInfo


logger = get_logger(__name__)


async def convert_mcp_prompt(client: MCPClient, prompt: MCPPrompt) -> StaticPrompt:
    """Convert MCP prompt to StaticPrompt."""
    from mcp.types import EmbeddedResource, ImageContent

    result = await client.get_prompt(prompt.name)
    return StaticPrompt(
        name=prompt.name,
        description=prompt.description or "No description provided",
        messages=[
            PromptMessage(role="system", content=message.content.text)
            for message in result.messages
            if not isinstance(message.content, EmbeddedResource | ImageContent)
        ],
    )


async def convert_mcp_resource(resource: MCPResource) -> ResourceInfo:
    """Convert MCP resource to ResourceInfo."""
    return ResourceInfo(
        name=resource.name, uri=str(resource.uri), description=resource.description
    )


class MCPManager(ResourceProvider):
    """Manages MCP server connections and tools."""

    def __init__(
        self,
        servers: Sequence[MCPServerConfig | str] | None = None,
        context: NodeContext | None = None,
    ):
        self.servers: list[MCPServerConfig] = []
        for server in servers or []:
            self.add_server_config(server)
        self.context = context
        self.clients: dict[str, MCPClient] = {}
        self.exit_stack = AsyncExitStack()

    def add_server_config(self, server: MCPServerConfig | str) -> None:
        """Add a new MCP server to the manager."""
        server = StdioMCPServer.from_string(server) if isinstance(server, str) else server
        self.servers.append(server)

    def __repr__(self) -> str:
        return f"MCPManager({self.servers!r})"

    async def __aenter__(self) -> Self:
        try:
            # Setup directly provided servers
            for server in self.servers:
                await self.setup_server(server)

            # Setup servers from context if available
            if self.context and self.context.config and self.context.config.mcp_servers:
                for server in self.context.config.get_mcp_servers():
                    await self.setup_server(server)

        except Exception as e:
            # Clean up in case of error
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize MCP manager"
            raise RuntimeError(msg) from e

        return self

    async def __aexit__(self, *exc):
        await self.cleanup()

    async def setup_server(self, config: MCPServerConfig) -> None:
        """Set up a single MCP server connection."""
        if not config.enabled:
            return
        env = config.get_env_vars()
        match config:
            case StdioMCPServer():
                client = MCPClient(stdio_mode=True)
                client = await self.exit_stack.enter_async_context(client)
                await client.connect(config.command, args=config.args, env=env)
                client_id = f"{config.command}_{' '.join(config.args)}"
            case SSEMCPServer():
                client = MCPClient(stdio_mode=False)
                client = await self.exit_stack.enter_async_context(client)
                await client.connect("", [], url=config.url, env=env)
                client_id = f"sse_{config.url}"

        self.clients[client_id] = client

    async def get_tools(self) -> list[ToolInfo]:
        """Get all tools from all connected servers."""
        from llmling_agent.tools.base import ToolInfo

        tools: list[ToolInfo] = []
        for client in self.clients.values():
            try:
                for tool in client._available_tools:
                    try:
                        fn = client.create_tool_callable(tool)
                        llm_tool = LLMCallableTool.from_callable(fn)
                        meta = {"mcp_tool": tool.name}
                        tool_info = ToolInfo(llm_tool, source="mcp", metadata=meta)
                        tools.append(tool_info)
                        logger.debug("Registered MCP tool: %s", tool.name)
                    except Exception:
                        msg = "Failed to create tool from MCP tool: %s"
                        logger.exception(msg, tool.name)
                        continue
            except Exception:
                client_id = next(k for k, v in self.clients.items() if v == client)
                logger.exception("Error getting tools from MCP server: %s", client_id)
                continue

        return tools

    async def list_prompts(self) -> list[StaticPrompt]:
        """Get all available prompts from MCP servers."""
        prompts = []
        for client in self.clients.values():
            try:
                result = await client.list_prompts()
                for prompt in result.prompts:
                    try:
                        converted = await convert_mcp_prompt(client, prompt)
                        prompts.append(converted)
                    except Exception:
                        logger.exception("Failed to convert prompt: %s", prompt.name)
            except Exception:
                logger.exception("Failed to get prompts from MCP server")
        return prompts

    async def list_resources(self) -> list[ResourceInfo]:
        """Get all available resources from MCP servers."""
        resources = []
        for client in self.clients.values():
            try:
                result = await client.list_resources()
                for resource in result.resources:
                    try:
                        converted = await convert_mcp_resource(resource)
                        resources.append(converted)
                    except Exception:
                        logger.exception("Failed to convert resource: %s", resource.name)
            except Exception:
                logger.exception("Failed to get resources from MCP server")
        return resources

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
