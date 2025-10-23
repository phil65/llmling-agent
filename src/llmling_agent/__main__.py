"""CLI interface for LLMling agents."""

from __future__ import annotations

from llmling.cli.utils import get_command_help
import typer as t

from llmling_agent_cli.agent import add_agent_file, list_agents, set_active_file
from llmling_agent_cli.chat import chat_command
from llmling_agent_cli.history import history_cli
from llmling_agent_cli.quickstart import quickstart_command
from llmling_agent_cli.run import run_command
from llmling_agent_cli.serve_acp import acp_command
from llmling_agent_cli.serve_api import api_command
from llmling_agent_cli.serve_mcp import serve_command
from llmling_agent_cli.task import task_command
from llmling_agent_cli.watch import watch_command


MAIN_HELP = "🤖 LLMling Agent CLI - Run and manage LLM agents"

# Create CLI app
help_text = get_command_help(MAIN_HELP)
cli = t.Typer(name="LLMling Agent", help=help_text, no_args_is_help=True)

cli.command(name="add")(add_agent_file)
cli.command(name="run")(run_command)
cli.command(name="list")(list_agents)
cli.command(name="set")(set_active_file)
cli.command(name="chat")(chat_command)
cli.command(name="quickstart")(quickstart_command)
cli.command(name="watch")(watch_command)
cli.command(name="serve-acp")(acp_command)
cli.command(name="serve-mcp")(serve_command)
cli.command(name="serve-api")(api_command)
cli.command(name="task")(task_command)

cli.add_typer(history_cli, name="history")
