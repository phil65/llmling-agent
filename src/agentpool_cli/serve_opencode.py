"""Command for running agents as an OpenCode-compatible server.

This creates an HTTP server that implements the OpenCode API protocol,
allowing OpenCode TUI and SDK clients to interact with AgentPool agents.
"""

from __future__ import annotations

from typing import Annotated

from platformdirs import user_log_path
import typer as t

from agentpool_cli import log, resolve_agent_config


logger = log.get_logger(__name__)


def opencode_command(
    config: Annotated[str | None, t.Argument(help="Path to agent configuration (optional)")] = None,
    host: Annotated[
        str,
        t.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        t.Option("--port", "-p", help="Port to listen on"),
    ] = 4096,
    agent: Annotated[
        str | None,
        t.Option(
            "--agent",
            help="Name of specific agent to use (defaults to first agent in config)",
        ),
    ] = None,
    working_dir: Annotated[
        str | None,
        t.Option(
            "--working-dir",
            "-w",
            help="Working directory for file operations (defaults to current directory)",
        ),
    ] = None,
) -> None:
    """Run agents as an OpenCode-compatible HTTP server.

    This creates an HTTP server implementing the OpenCode API protocol,
    enabling your AgentPool agents to work with OpenCode TUI and SDK clients.

    Configuration:
    Config file is optional. Without a config file, creates a general-purpose
    agent with default settings similar to the ACP server.

    Agent Selection:
    Use --agent to specify which agent to use by name. Without this option,
    the first agent in your config is used as the default (or "assistant"
    if no config provided).

    Examples:
        # Start with default agent
        agentpool serve-opencode

        # Start with specific config
        agentpool serve-opencode agents.yml

        # Start on custom port with specific agent
        agentpool serve-opencode --port 8080 --agent myagent agents.yml
    """
    from agentpool import AgentPool, log as ap_log
    from agentpool.config_resources import CLAUDE_CODE_ASSISTANT
    from agentpool_server.opencode_server import OpenCodeServer

    # Always log to file with rollover
    log_dir = user_log_path("agentpool", appauthor=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "opencode.log"
    ap_log.configure_logging(force=True, log_file=str(log_file))
    logger.info("Configured file logging with rollover", log_file=str(log_file))

    # Resolve config path (use default if not provided)
    if config:
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            raise t.BadParameter(str(e)) from e
    else:
        config_path = CLAUDE_CODE_ASSISTANT

    logger.info("Starting OpenCode server", config_path=config_path, host=host, port=port)

    # Load agent from config
    pool = AgentPool(config_path)

    async def run_server() -> None:
        async with pool:
            server = OpenCodeServer(
                pool,
                host=host,
                port=port,
                working_dir=working_dir,
                agent_name=agent,
            )
            logger.info("Server starting", url=f"http://{host}:{port}")
            await server.run_async()

    import asyncio

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("OpenCode server shutdown requested")
    except Exception as e:
        logger.exception("OpenCode server error")
        raise t.Exit(1) from e


if __name__ == "__main__":
    t.run(opencode_command)
