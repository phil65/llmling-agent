"""Provider for integration tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import HttpUrl

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig


async def add_local_mcp_server(  # noqa: D417
    ctx: AgentContext,
    name: str,
    command: str,
    args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Add a local MCP server via stdio transport.

    Args:
        name: Unique name for the MCP server
        command: Command to execute for the server
        args: Command arguments
        env_vars: Environment variables to pass to the server

    Returns:
        Confirmation message about the added server
    """
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    env = env_vars or {}
    config = StdioMCPServerConfig(name=name, command=command, args=args or [], env=env)
    await ctx.agent.mcp.setup_server_runtime(config)
    # New provider automatically available via aggregating provider
    return f"Added local MCP server **{name}** with command: **{command}**"


async def add_remote_mcp_server(  # noqa: D417
    ctx: AgentContext,
    name: str,
    url: str,
    transport: Literal["sse", "streamable-http"] = "streamable-http",
) -> str:
    """Add a remote MCP server via HTTP-based transport.

    Args:
        name: Unique name for the MCP server
        url: Server URL endpoint
        transport: HTTP transport type to use (http is preferred)

    Returns:
        Confirmation message about the added server
    """
    from llmling_agent_config.mcp_server import SSEMCPServerConfig, StreamableHTTPMCPServerConfig

    match transport:
        case "sse":
            config: MCPServerConfig = SSEMCPServerConfig(name=name, url=HttpUrl(url))
        case "streamable-http":
            config = StreamableHTTPMCPServerConfig(name=name, url=HttpUrl(url))

    await ctx.agent.mcp.setup_server_runtime(config)
    # New provider automatically available via aggregating provider
    return f"Added remote MCP server **{name}** at *{url}* using {transport} transport"


class IntegrationTools(ResourceProvider):
    """Provider for integration tools."""

    def __init__(self, name: str = "integrations") -> None:
        super().__init__(name)

    async def get_tools(self) -> list[Tool]:
        """Get integration tools."""
        return [
            self.create_tool(add_local_mcp_server, category="other"),
            self.create_tool(add_remote_mcp_server, category="other", open_world=True),
        ]
