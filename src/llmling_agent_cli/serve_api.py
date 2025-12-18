"""Command for running agents as a completions API server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config
from llmling_agent_server.openai_api_server.server import OpenAIAPIServer


if TYPE_CHECKING:
    from llmling_agent import ChatMessage


logger = get_logger(__name__)


def api_command(
    ctx: t.Context,
    config: Annotated[str | None, t.Argument(help="Path to agent configuration")] = None,
    host: Annotated[str, t.Option(help="Host to bind server to")] = "localhost",
    port: Annotated[int, t.Option(help="Port to listen on")] = 8000,
    cors: Annotated[bool, t.Option(help="Enable CORS")] = True,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity")
    ] = False,
    docs: Annotated[bool, t.Option(help="Enable API documentation")] = True,
) -> None:
    """Run agents as a completions API server.

    This creates an OpenAI-compatible API server that makes your agents available
    through a standard completions API interface.
    """
    import uvicorn

    from llmling_agent import AgentPool, AgentsManifest

    def on_message(message: ChatMessage[Any]) -> None:
        print(message.format(style="simple"))

    try:
        config_path = resolve_agent_config(config)
    except ValueError as e:
        msg = str(e)
        raise t.BadParameter(msg) from e
    manifest = AgentsManifest.from_file(config_path)
    pool = AgentPool(manifest)

    if show_messages:
        for agent in pool.agents.values():
            agent.message_sent.connect(on_message)

    server = OpenAIAPIServer(pool, cors=cors, docs=docs)

    # Get log level from the global context
    log_level = ctx.obj.get("log_level", "info") if ctx.obj else "info"
    uvicorn.run(server.app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    import typer

    typer.run(api_command)
