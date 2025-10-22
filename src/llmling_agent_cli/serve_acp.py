"""Command for running agents as an ACP (Agent Client Protocol) server.

This creates an ACP-compatible JSON-RPC 2.0 server that exposes your agents
for bidirectional communication over stdio streams, enabling desktop application
integration with file system access, permission handling, and terminal support.
"""

from __future__ import annotations

import asyncio
import logging

import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config
from llmling_agent_cli.cli_types import LogLevel  # noqa: TC001


logger = get_logger(__name__)


def acp_command(
    config: str = t.Argument(None, help="Path to agent configuration"),
    log_level: LogLevel = t.Option("info", help="Logging level"),  # noqa: B008
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
    session_support: bool = t.Option(
        True,
        "--session-support/--no-session-support",
        help="Enable session loading support",
    ),
    show_messages: bool = t.Option(
        False, "--show-messages", help="Show message activity in logs"
    ),
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
):
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

    Agent Selection:
    Use --agent to specify which agent to use by name. Without this option,
    the first agent in your config is used as the default.

    Agent Mode Switching:
    If your config defines multiple agents, the IDE will show a mode selector
    allowing users to switch between agents mid-conversation. Each agent appears
    as a different "mode" with its own name and capabilities.

    Examples:
        # Run ACP server with single agent
        llmling-agent acp config.yml

        # Run with specific agent by name
        llmling-agent acp config.yml --agent my-agent

        # Run with multiple agents (enables mode switching)
        llmling-agent acp multi-agent-config.yml

        # Run with file system access enabled
        llmling-agent acp config.yml --file-access

        # Run with full capabilities
        llmling-agent acp config.yml --file-access --terminal-access

        # Run with debug commands for testing
        llmling-agent acp config.yml --debug-commands

        # Test with ACP client (example)
        echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}' | python -m llmling_agent_cli acp config.yml

    Protocol Flow:
        1. Client sends initialize request
        2. Server responds with capabilities
        3. Client creates new session with available agent modes
        4. User can switch modes (agents) via IDE interface
        5. Client sends prompt requests
        6. Server streams responses via session updates
    """  # noqa: E501
    from llmling_agent_acp import ACPServer

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    async def run_acp_server():
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        logger.info("Starting ACP server for config: %s", config_path)

        # Create ACP server from config
        try:
            acp_server = await ACPServer.from_config(
                config_path,
                session_support=session_support,
                file_access=file_access,
                terminal_access=terminal_access,
                providers=providers,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
                debug_messages=debug_messages,
                debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
                debug_commands=debug_commands,
                agent=agent,
            )
        except Exception as e:
            logger.exception("Failed to create ACP server from config")
            raise t.Exit(1) from e

        # Configure agent capabilities
        agent_count = len(acp_server.agent_pool.agents)
        if agent_count == 0:
            logger.error("No agents found in configuration")
            raise t.Exit(1)
        logger.info("Configured %d agents for ACP protocol", agent_count)
        if show_messages:
            logger.info("Message activity logging enabled")
        if debug_messages:
            debug_path = debug_file or "acp-debug.jsonl"
            logger.info("Raw JSON-RPC message debugging enabled -> %s", debug_path)
        if debug_commands:
            logger.info("Debug slash commands enabled")
        msg = "Starting ACP server (file_access=%s terminal_access=%s session_support=%s)"
        logger.info(msg, file_access, terminal_access, session_support)

        try:
            await acp_server.run()
        except KeyboardInterrupt:
            logger.info("ACP server shutdown requested")
        except Exception as e:
            logger.exception("ACP server error")
            raise t.Exit(1) from e
        finally:
            await acp_server.shutdown()

    try:
        asyncio.run(run_acp_server())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception("Failed to run ACP server")
        raise t.Exit(1) from e


def acp_debug_command(
    sequence_file: str | None = t.Option(
        None, "--sequence", help="Path to JSON file containing notification sequence"
    ),
    create_sample: str | None = t.Option(
        None, "--create-sample", help="Create sample sequence file at specified path"
    ),
    log_level: LogLevel = t.Option("info", help="Logging level"),  # noqa: B008
):
    """Run ACP debug server for testing client behavior.

    This is a minimal ACP server that can send predefined notification sequences
    to test how clients handle various ACP messages without needing a full agent setup.

    Examples:
        # Run debug server with default sequence
        llmling-agent acp-debug

        # Create a sample sequence file
        llmling-agent acp-debug --create-sample debug_sequence.json

        # Run with custom sequence
        llmling-agent acp-debug --sequence debug_sequence.json

        # Test with ACP client
        echo '{"jsonrpc":"2.0","method":"initialize",
        "params":{"protocolVersion":1},"id":1}' | llmling-agent acp-debug
    """
    import json
    from pathlib import Path

    from llmling_agent_acp.debug_server import DebugServer, create_sample_sequence

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    if create_sample:
        sample_path = Path(create_sample)
        sample_sequence = create_sample_sequence()

        with sample_path.open("w") as f:
            json.dump(sample_sequence, f, indent=2)

        print(f"Created sample sequence file: {sample_path}")
        return

    async def run_debug_server():
        logger.info("Starting ACP debug server")
        server = DebugServer(sequence_file=sequence_file)
        await server.run()

    try:
        asyncio.run(run_debug_server())
    except KeyboardInterrupt:
        logger.info("Debug server interrupted")
    except Exception as e:
        logger.exception("Failed to run debug server")
        raise t.Exit(1) from e


if __name__ == "__main__":
    t.run(acp_command)
