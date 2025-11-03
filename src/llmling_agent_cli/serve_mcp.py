"""Command for running agents as an MCP server."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import typer as t

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent.log import get_logger
from llmling_agent_cli.cli_types import LogLevel  # noqa: TC001
from llmling_agent_config.pool_server import (
    MCPPoolServerConfig,
    TransportType,  # noqa: TC001
)


if TYPE_CHECKING:
    from llmling_agent import ChatMessage


logger = get_logger(__name__)


def serve_command(
    config: str = t.Argument(..., help="Path to agent configuration"),
    transport: TransportType = t.Option("stdio", help="Transport type"),  # noqa: B008
    host: str = t.Option(
        "localhost", help="Host to bind server to (sse/streamable-http only)"
    ),
    port: int = t.Option(3001, help="Port to listen on (sse/streamable-http only)"),
    zed_mode: bool = t.Option(False, help="Enable Zed editor compatibility"),
    show_messages: bool = t.Option(
        False, "--show-messages", help="Show message activity"
    ),
    log_level: LogLevel = t.Option("info", help="Logging level"),  # noqa: B008
):
    """Run agents as an MCP server.

    This makes agents available as tools to other applications, regardless of
    whether pool_server is configured in the manifest.
    """
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    def on_message(message: ChatMessage[Any]):
        print(message.format(style="simple"))

    async def run_server():
        # Override/set server config before creating pool
        manifest = AgentsManifest.from_file(config)
        manifest.pool_server = MCPPoolServerConfig(
            enabled=True,
            transport=transport,
            host=host,
            port=port,
            zed_mode=zed_mode,
        )
        async with AgentPool(manifest) as pool:
            if show_messages:
                for agent in pool.agents.values():
                    agent.message_sent.connect(on_message)

            try:
                await pool.run_event_loop()
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")

    asyncio.run(run_server())
