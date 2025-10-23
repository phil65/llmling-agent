"""Command for running agents as a completions API server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config
from llmling_agent_cli.cli_types import LogLevel  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent import ChatMessage


logger = get_logger(__name__)


def api_command(
    config: str = t.Argument(None, help="Path to agent configuration"),
    host: str = t.Option("localhost", help="Host to bind server to"),
    port: int = t.Option(8000, help="Port to listen on"),
    cors: bool = t.Option(True, help="Enable CORS"),
    show_messages: bool = t.Option(
        False, "--show-messages", help="Show message activity"
    ),
    log_level: LogLevel = t.Option("info", help="Logging level"),  # noqa: B008
    docs: bool = t.Option(True, help="Enable API documentation"),
):
    """Run agents as a completions API server.

    This creates an OpenAI-compatible API server that makes your agents available
    through a standard completions API interface.

    Examples:
        # Run API server on default port
        llmling-agent api config.yml

        # Run on custom port with CORS disabled
        llmling-agent api config.yml --port 8080 --no-cors
    """
    import uvicorn

    from llmling_agent import AgentPool, AgentsManifest
    from llmling_agent_server.completions.server import OpenAIServer

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    def on_message(message: ChatMessage[Any]):
        print(message.format(style="simple"))

    try:
        config_path = resolve_agent_config(config)
    except ValueError as e:
        msg = str(e)
        raise t.BadParameter(msg) from e
    manifest = AgentsManifest.from_file(config_path)
    pool = AgentPool[None](manifest)

    if show_messages:
        for agent in pool.agents.values():
            agent.message_sent.connect(on_message)

    server = OpenAIServer(pool, cors=cors, docs=docs)

    uvicorn.run(server.app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    import typer

    typer.run(api_command)
