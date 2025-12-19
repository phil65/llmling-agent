"""CLI interface for LLMling agents."""

from __future__ import annotations

import typer as t

from llmling_agent import log
from llmling_agent_cli.agent import add_agent_file, list_agents, set_active_file
from llmling_agent_cli.cli_types import LogLevel  # noqa: TC001
from llmling_agent_cli.history import history_cli
from llmling_agent_cli.run import run_command
from llmling_agent_cli.serve_acp import acp_command
from llmling_agent_cli.serve_api import api_command
from llmling_agent_cli.serve_mcp import serve_command
from llmling_agent_cli.serve_vercel import vercel_command
from llmling_agent_cli.store import config_store
from llmling_agent_cli.task import task_command
from llmling_agent_cli.watch import watch_command


MAIN_HELP = "🤖 LLMling Agent CLI - Run and manage LLM agents"


def get_command_help(base_help: str) -> str:
    """Get command help text with active config information."""
    if active := config_store.get_active():
        return f"{base_help}\n\n(Using config: {active})"
    return f"{base_help}\n\n(No active config set)"


def main(
    ctx: t.Context,
    log_level: LogLevel = t.Option("info", "--log-level", "-l", help="Log level"),  # noqa: B008
) -> None:
    """🤖 LLMling Agent CLI - Run and manage LLM agents."""
    # Configure logging globally
    log.configure_logging(level=log_level.upper())

    # Store log level in context for commands that might need it
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


# Create CLI app
help_text = get_command_help(MAIN_HELP)
cli = t.Typer(name="LLMling Agent", help=help_text, no_args_is_help=True)
cli.callback()(main)

cli.command(name="add")(add_agent_file)
cli.command(name="run")(run_command)
cli.command(name="list")(list_agents)
cli.command(name="set")(set_active_file)
cli.command(name="watch")(watch_command)
cli.command(name="serve-acp")(acp_command)
cli.command(name="serve-mcp")(serve_command)
cli.command(name="serve-api")(api_command)
cli.command(name="serve-vercel")(vercel_command)
cli.command(name="task")(task_command)

cli.add_typer(history_cli, name="history")


if __name__ == "__main__":
    cli()
