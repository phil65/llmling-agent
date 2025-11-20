"""Command for running agents as an ACP (Agent Client Protocol) server.

This creates an ACP-compatible JSON-RPC 2.0 server that exposes your agents
for bidirectional communication over stdio streams, enabling desktop application
integration with file system access, permission handling, and terminal support.
"""

from __future__ import annotations

import asyncio

import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config


logger = get_logger(__name__)


def acp_command(
    config: str | None = t.Argument(None, help="Path to agent configuration (optional)"),
    file_access: bool = t.Option(
        True,
        "--file-access/--no-file-access",
        help="Enable file system access for agents",
    ),
    terminal_access: bool = t.Option(
        True,
        "--terminal-access/--no-terminal-access",
        help="Enable terminal access for agents",
    ),
    show_messages: bool = t.Option(False, "--show-messages", help="Show message activity in logs"),
    debug_messages: bool = t.Option(
        False, "--debug-messages", help="Save raw JSON-RPC messages to debug file"
    ),
    debug_file: str | None = t.Option(
        None,
        "--debug-file",
        help="File to save JSON-RPC debug messages (default: acp-debug.jsonl)",
    ),
    providers: list[str] | None = t.Option(  # noqa: B008
        None,
        "--model-provider",
        help="Providers to search for models (can be specified multiple times)",
    ),
    debug_commands: bool = t.Option(
        False,
        "--debug-commands",
        help="Enable debug slash commands for testing ACP notifications",
    ),
    agent: str | None = t.Option(
        None,
        "--agent",
        help="Name of specific agent to use (defaults to first agent in config)",
    ),
) -> None:
    r"""Run agents as an ACP (Agent Client Protocol) server.

    This creates an ACP-compatible JSON-RPC 2.0 server that communicates over stdio
    streams, enabling your agents to work with desktop applications that support
    the Agent Client Protocol.

    The ACP protocol provides:
    - Bidirectional JSON-RPC 2.0 communication
    - Session management and conversation history
    - Agent switching via session modes (if multiple agents configured)
    - File system operations with permission handling
    - Terminal integration (optional)
    - Content blocks (text, image, audio, resources)
    - Debug slash commands for testing ACP notifications (optional)

    Configuration:
    Config file is optional. Without a config file, creates a general-purpose
    agent with default settings. This is useful for clients/installers that
    start agents directly without configuration support.

    Agent Selection:
    Use --agent to specify which agent to use by name. Without this option,
    the first agent in your config is used as the default (or "llmling-agent"
    if no config provided).

    Agent Mode Switching:
    If your config defines multiple agents, the IDE will show a mode selector
    allowing users to switch between agents mid-conversation. Each agent appears
    as a different "mode" with its own name and capabilities.

    Examples:
        # Run ACP server with config file
        llmling-agent serve-acp config.yml

        # Run without config (minimal setup)
        llmling-agent serve-acp

        # Run without config with custom agent name
        llmling-agent serve-acp --agent my-assistant

        # Run with specific agent by name
        llmling-agent serve-acp config.yml --agent my-agent

        # Run with multiple agents (enables mode switching)
        llmling-agent serve-acp multi-agent-config.yml

        # Run with file system access enabled
        llmling-agent serve-acp config.yml --file-access

        # Run with full capabilities
        llmling-agent serve-acp config.yml --file-access --terminal-access

        # Run with debug commands for testing
        llmling-agent serve-acp config.yml --debug-commands

        # Test with ACP client (example)
        echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}' | llmling-agent serve-acp

    Protocol Flow:
        1. Client sends initialize request
        2. Server responds with capabilities
        3. Client creates new session with available agent modes
        4. User can switch modes (agents) via IDE interface
        5. Client sends prompt requests
        6. Server streams responses via session updates
    """  # noqa: E501
    from llmling_agent_server.acp_server import ACPServer

    if config:
        # Use config file
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        logger.info("Starting ACP server", config_path=config_path)
        acp_server = ACPServer.from_config(
            config_path,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
        )
    else:
        # Create minimal pool without config
        from llmling_agent import Agent
        from llmling_agent.delegation import AgentPool

        logger.info("Starting ACP server with minimal configuration")

        # Create a simple general-purpose agent
        agent_name = agent or "llmling-agent"
        default_agent = Agent(
            name=agent_name,
            description="A general-purpose AI assistant",
            system_prompt=[
                "You are a helpful AI assistant that can help with various tasks.",
                "You have access to file operations and can assist with coding, writing,",
                " analysis, and more.",
                "Be concise but thorough in your responses.",
            ],
        )

        # Create pool with the agent
        pool = AgentPool()
        pool.register(agent_name, default_agent)

        acp_server = ACPServer(
            pool,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent_name,
        )
    # Configure agent capabilities
    agent_count = len(acp_server.pool.agents)
    if agent_count == 0:
        logger.error("No agents found in configuration")
        raise t.Exit(1)
    logger.info("Configured %d agents for ACP protocol", agent_count)
    if show_messages:
        logger.info("Message activity logging enabled")
    if debug_messages:
        debug_path = debug_file or "acp-debug.jsonl"
        logger.info("Raw JSON-RPC message debugging enabled", path=debug_path)
    if debug_commands:
        logger.info("Debug slash commands enabled")

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
