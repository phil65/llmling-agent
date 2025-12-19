"""Command for running agents as an MCP server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any

import typer as t

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent.log import get_logger
from llmling_agent_config.pool_server import (
    MCPPoolServerConfig,
    TransportType,  # noqa: TC001
)


if TYPE_CHECKING:
    from llmling_agent import ChatMessage


logger = get_logger(__name__)


def serve_command(
    config: Annotated[str, t.Argument(help="Path to agent configuration")],
    transport: Annotated[TransportType, t.Option(help="Transport type")] = "stdio",
    host: Annotated[
        str, t.Option(help="Host to bind server to (sse/streamable-http only)")
    ] = "localhost",
    port: Annotated[int, t.Option(help="Port to listen on (sse/streamable-http only)")] = 3001,
    zed_mode: Annotated[bool, t.Option(help="Enable Zed editor compatibility")] = False,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity")
    ] = False,
) -> None:
    """Run agents as an MCP server.

    This makes agents available as tools to other applications, regardless of
    whether pool_server is configured in the manifest.
    """

    def on_message(message: ChatMessage[Any]) -> None:
        print(message.format(style="simple"))

    async def run_server() -> None:
        # Load manifest and create pool (without server config)
        manifest = AgentsManifest.from_file(config)
        pool = AgentPool(manifest)

        # Create server config and server externally
        server_config = MCPPoolServerConfig(
            enabled=True,
            transport=transport,
            host=host,
            port=port,
            zed_mode=zed_mode,
        )

        from llmling_agent_server.mcp_server.server import MCPServer

        server = MCPServer(pool, server_config)

        async with pool, server:
            if show_messages:
                for agent in pool.agents.values():
                    agent.message_sent.connect(on_message)

            try:
                await server.start()  # Blocks until server stops
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")

    asyncio.run(run_server())
