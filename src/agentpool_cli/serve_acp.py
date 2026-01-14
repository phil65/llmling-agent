"""Command for running agents as an ACP (Agent Client Protocol) server.

This creates an ACP-compatible JSON-RPC 2.0 server that exposes your agents
for bidirectional communication over stdio streams, enabling desktop application
integration with file system access, permission handling, and terminal support.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Annotated, Literal

from platformdirs import user_log_path
import typer as t

from agentpool_cli import log, resolve_agent_config


if TYPE_CHECKING:
    from acp import Transport
    from agentpool_config.mcp_server import MCPServerConfig


logger = log.get_logger(__name__)


def acp_command(  # noqa: PLR0915
    # Too many statements - complex CLI command with many options
    config: Annotated[str | None, t.Argument(help="Path to agent configuration (optional)")] = None,
    file_access: Annotated[
        bool,
        t.Option(
            "--file-access/--no-file-access",
            help="Enable file system access for agents",
        ),
    ] = True,
    terminal_access: Annotated[
        bool,
        t.Option(
            "--terminal-access/--no-terminal-access",
            help="Enable terminal access for agents",
        ),
    ] = True,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity in logs")
    ] = False,
    debug_messages: Annotated[
        bool, t.Option("--debug-messages", help="Save raw JSON-RPC messages to debug file")
    ] = False,
    debug_file: Annotated[
        str | None,
        t.Option(
            "--debug-file",
            help="File to save JSON-RPC debug messages (default: acp-debug.jsonl)",
        ),
    ] = None,
    debug_commands: Annotated[
        bool,
        t.Option(
            "--debug-commands",
            help="Enable debug slash commands for testing ACP notifications",
        ),
    ] = False,
    agent: Annotated[
        str | None,
        t.Option(
            "--agent",
            help="Name of specific agent to use (defaults to first agent in config)",
        ),
    ] = None,
    load_skills: Annotated[
        bool,
        t.Option(
            "--skills/--no-skills",
            help="Load client-side skills from .claude/skills directory",
        ),
    ] = True,
    transport: Annotated[
        Literal["stdio", "websocket"],
        t.Option(
            "--transport",
            "-t",
            help="Transport type: stdio (default) or websocket",
        ),
    ] = "stdio",
    ws_host: Annotated[
        str,
        t.Option(
            "--ws-host",
            help="WebSocket host (only used with --transport websocket)",
        ),
    ] = "localhost",
    ws_port: Annotated[
        int,
        t.Option(
            "--ws-port",
            help="WebSocket port (only used with --transport websocket)",
        ),
    ] = 8765,
    mcp_config: Annotated[
        str | None,
        t.Option(
            "--mcp-config",
            help='MCP servers configuration as JSON (format: {"mcpServers": {...}})',
        ),
    ] = None,
) -> None:
    r"""Run agents as an ACP (Agent Client Protocol) server.

    This creates an ACP-compatible JSON-RPC 2.0 server that communicates over stdio
    streams, enabling your agents to work with desktop applications that support
    the Agent Client Protocol.

    Configuration:
    Config file is optional. Without a config file, creates a general-purpose
    agent with default settings. This is useful for clients/installers that
    start agents directly without configuration support.

    Agent Selection:
    Use --agent to specify which agent to use by name. Without this option,
    the first agent in your config is used as the default (or "agentpool"
    if no config provided).

    Agent Mode Switching:
    If your config defines multiple agents, the IDE will show a mode selector
    allowing users to switch between agents mid-conversation. Each agent appears
    as a different "mode" with its own name and capabilities.
    """
    from acp import StdioTransport, WebSocketTransport
    from agentpool import log
    from agentpool.config_resources import ACP_ASSISTANT
    from agentpool_server.acp_server import ACPServer

    # Build transport config
    if transport == "websocket":
        transport_config: Transport = WebSocketTransport(host=ws_host, port=ws_port)
    elif transport == "stdio":
        transport_config = StdioTransport()
    else:
        raise t.BadParameter(f"Unknown transport: {transport}. Use 'stdio' or 'websocket'.")

    # Always log to file with rollover
    log_dir = user_log_path("agentpool", appauthor=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "acp.log"
    log.configure_logging(force=True, log_file=str(log_file))
    logger.info("Configured file logging with rollover", log_file=str(log_file))

    if config:
        # Use config file
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        logger.info("Starting ACP server", config_path=config_path, transport=transport)
        acp_server = ACPServer.from_config(
            config_path,
            file_access=file_access,
            terminal_access=terminal_access,
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
            load_skills=load_skills,
            transport=transport_config,
        )
    else:
        # Use default ACP assistant config
        logger.info("Starting ACP server with default configuration", transport=transport)
        acp_server = ACPServer.from_config(
            ACP_ASSISTANT,
            file_access=file_access,
            terminal_access=terminal_access,
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
            load_skills=load_skills,
            transport=transport_config,
        )

    # Inject MCP servers from --mcp-config if provided
    # TODO: Consider adding to specific agent's MCP manager instead of pool-level
    # for better isolation (currently all agents in pool share these servers)
    if mcp_config:
        try:
            mcp_data = json.loads(mcp_config)
            if "mcpServers" not in mcp_data:
                raise t.BadParameter("MCP config must contain 'mcpServers' key")

            from agentpool_config.mcp_server import (
                SSEMCPServerConfig,
                StreamableHTTPMCPServerConfig,
            )

            for server_name, server_cfg in mcp_data["mcpServers"].items():
                # Parse server config based on transport type
                if "transport" in server_cfg:
                    if server_cfg["transport"] == "sse":
                        server: MCPServerConfig = SSEMCPServerConfig(
                            name=server_name,
                            url=server_cfg["url"],
                        )
                    elif server_cfg["transport"] == "http":
                        server = StreamableHTTPMCPServerConfig(
                            name=server_name,
                            url=server_cfg["url"],
                        )
                    else:
                        msg = f"Unsupported transport type: {server_cfg['transport']}"
                        raise t.BadParameter(msg)
                else:
                    # Default to HTTP if no transport specified
                    server = StreamableHTTPMCPServerConfig(
                        name=server_name,
                        url=server_cfg["url"],
                    )

                acp_server.pool.mcp.add_server_config(server)
                logger.info(
                    "Added MCP server from --mcp-config",
                    server_name=server_name,
                    url=server_cfg.get("url"),
                )
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in --mcp-config: {e}"
            raise t.BadParameter(msg) from e

    if show_messages:
        logger.info("Message activity logging enabled")
    if debug_messages:
        debug_path = debug_file or "acp-debug.jsonl"
        logger.info("Raw JSON-RPC message debugging enabled", path=debug_path)
    if debug_commands:
        logger.info("Debug slash commands enabled")
    logger.info("Server PID", pid=os.getpid())

    async def run_acp_server() -> None:
        try:
            async with acp_server:
                await acp_server.start()
        except KeyboardInterrupt:
            logger.info("ACP server shutdown requested")
        except Exception as e:
            logger.exception("ACP server error")
            raise t.Exit(1) from e

    asyncio.run(run_acp_server())


if __name__ == "__main__":
    t.run(acp_command)
