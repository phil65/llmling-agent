"""Command for running agents as an ACP (Agent Client Protocol) server.

This creates an ACP-compatible JSON-RPC 2.0 server that exposes your agents
for bidirectional communication over stdio streams, enabling desktop application
integration with file system access, permission handling, and terminal support.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from platformdirs import user_log_path
import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config


logger = get_logger(__name__)


def acp_command(
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
    providers: Annotated[
        list[str] | None,
        t.Option(
            "--model-provider",
            help="Providers to search for models (can be specified multiple times)",
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
    the first agent in your config is used as the default (or "llmling-agent"
    if no config provided).

    Agent Mode Switching:
    If your config defines multiple agents, the IDE will show a mode selector
    allowing users to switch between agents mid-conversation. Each agent appears
    as a different "mode" with its own name and capabilities.
    """
    from llmling_agent import log
    from llmling_agent_server.acp_server import ACPServer

    # Always log to file with rollover
    log_dir = user_log_path("llmling-agent", appauthor=False)
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

        logger.info("Starting ACP server", config_path=config_path)
        acp_server = ACPServer.from_config(
            config_path,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,  # type: ignore[arg-type]
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
            load_skills=load_skills,
        )
    else:
        # Use default ACP assistant config
        from llmling_agent.config_resources import ACP_ASSISTANT

        logger.info("Starting ACP server with default configuration")
        acp_server = ACPServer.from_config(
            ACP_ASSISTANT,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,  # type: ignore[arg-type]
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
            load_skills=load_skills,
        )
    # Configure agent capabilities
    agent_count = len(acp_server.pool.all_agents)
    if agent_count == 0:
        logger.error("No agents found in configuration")
        raise t.Exit(1)
    logger.info("Configured agents for ACP protocol", count=agent_count)
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
