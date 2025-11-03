"""History management commands."""

from __future__ import annotations

import asyncio
from datetime import datetime

from llmling.cli.constants import output_format_opt
from llmling.cli.utils import OutputFormat, format_output
import typer as t

from llmling_agent import AgentsManifest, log
from llmling_agent.utils.now import get_now
from llmling_agent.utils.parse_time import parse_time_period
from llmling_agent_cli import resolve_agent_config
from llmling_agent_cli.cli_types import GroupBy


logger = log.get_logger(__name__)

help_text = "Conversation history management"
history_cli = t.Typer(name="history", help=help_text, no_args_is_help=True)

AGENT_NAME_HELP = "Agent name (shows all if not provided)"
SINCE_HELP = "Show conversations since (YYYY-MM-DD or YYYY-MM-DD HH:MM)"
PERIOD_HELP = "Show conversations from last period (1h, 2d, 1w, 1m)"
COMPACT_HELP = "Show only first/last message of conversations"
TOKEN_HELP = "Include token usage statistics"
CONFIG_HELP = "Override agent config path"


def get_history_provider(config_path: str):
    """Get history provider from manifest config.

    Args:
        config_path: Path to agent configuration file

    Returns:
        Storage provider configured for history operations
    """
    from llmling_agent.storage import StorageManager

    manifest = AgentsManifest.from_file(config_path)
    storage = StorageManager(manifest.storage)
    return storage.get_history_provider()


@history_cli.command(name="show")
def show_history(
    agent_name: str | None = t.Argument(None, help=AGENT_NAME_HELP),
    config: str | None = t.Option(None, "--config", "-c", help=CONFIG_HELP),
    # Time-based filtering
    since: datetime | None = t.Option(None, "--since", "-s", help=SINCE_HELP),  # noqa: B008
    period: str | None = t.Option(None, "--period", "-p", help=PERIOD_HELP),
    # Content filtering
    query: str | None = t.Option(None, "--query", "-q", help="Search in message content"),
    model: str | None = t.Option(None, "--model", "-m", help="Filter by model used"),
    # Output control
    limit: int = t.Option(10, "--limit", "-n", help="Number of conversations"),
    compact: bool = t.Option(False, "--compact", help=COMPACT_HELP),
    tokens: bool = t.Option(False, "--tokens", "-t", help=TOKEN_HELP),
    output_format: OutputFormat = output_format_opt,
):
    """Show conversation history with filtering options.

    Examples:
        # Show last 5 conversations
        llmling-agent history show -n 5

        # Show conversations from last 24 hours
        llmling-agent history show --period 24h

        # Show conversations since specific date
        llmling-agent history show --since 2024-01-01

        # Search for specific content
        llmling-agent history show --query "database schema"

        # Show GPT-5 conversations with token usage
        llmling-agent history show --model gpt-5 --tokens

        # Compact view of recent conversations
        llmling-agent history show --period 1d --compact
    """
    try:
        # Resolve config and get provider
        config_path = resolve_agent_config(config)
        provider = get_history_provider(config_path)

        async def main():
            async with provider:
                results = await provider.get_filtered_conversations(
                    agent_name=agent_name,
                    period=period,
                    since=since,
                    query=query,
                    model=model,
                    limit=limit,
                    compact=compact,
                    include_tokens=tokens,
                )
                format_output(results, output_format)

        asyncio.run(main())

    except Exception as e:
        logger.exception("Failed to show history")
        raise t.Exit(1) from e


@history_cli.command(name="stats")
def show_stats(
    agent_name: str | None = t.Argument(None, help=AGENT_NAME_HELP),
    config: str | None = t.Option(None, "--config", "-c", help=CONFIG_HELP),
    period: str = t.Option(
        "1d", "--period", "-p", help="Time period (1h, 1d, 1w, 1m, 1y)"
    ),
    group_by: GroupBy = t.Option("agent", "--group-by", "-g", help="Group by"),  # noqa: B008
    output_format: OutputFormat = output_format_opt,
):
    """Show usage statistics.

    Examples:
        # Show stats for all agents
        llmling-agent history stats

        # Show daily stats for specific agent
        llmling-agent history stats myagent --group-by day

        # Show model usage for last week
        llmling-agent history stats --period 1w --group-by model
    """
    from llmling_agent_storage.formatters import format_stats
    from llmling_agent_storage.models import StatsFilters

    try:
        # Resolve config and get provider
        config_path = resolve_agent_config(config)
        provider = get_history_provider(config_path)

        # Create filters
        cutoff = get_now() - parse_time_period(period)
        filters = StatsFilters(cutoff=cutoff, group_by=group_by, agent_name=agent_name)  # type: ignore

        async def main():
            async with provider:
                stats = await provider.get_conversation_stats(filters)
            formatted = format_stats(stats, period, group_by)
            format_output(formatted, output_format)

        asyncio.run(main())

    except Exception as e:
        logger.exception("Failed to show stats")
        raise t.Exit(1) from e


@history_cli.command(name="reset")
def reset_history(
    config: str | None = t.Option(None, "--config", "-c", help=CONFIG_HELP),
    confirm: bool = t.Option(False, "--confirm", "-y", help="Confirm deletion"),
    agent_name: str | None = t.Option(
        None, "--agent", "-a", help="Only delete for specific agent"
    ),
    hard: bool = t.Option(
        False, "--hard", help="Drop and recreate tables (for schema changes)"
    ),
):
    """Reset (clear) conversation history.

    Examples:
        # Clear all history (with confirmation)
        llmling-agent history reset

        # Clear without confirmation
        llmling-agent history reset --confirm

        # Clear history for specific agent
        llmling-agent history reset --agent myagent

        # Drop and recreate tables (for schema changes)
        llmling-agent history reset --hard --confirm
    """
    try:
        # Resolve config and get provider
        config_path = resolve_agent_config(config)
        provider = get_history_provider(config_path)

        if not confirm:
            what = f" for {agent_name}" if agent_name else ""
            msg = f"This will delete all history{what}. Are you sure? [y/N] "
            if input(msg).lower() != "y":
                print("Operation cancelled.")
                return

        async def main():
            async with provider:
                conv_count, msg_count = await provider.reset(
                    agent_name=agent_name, hard=hard
                )

                what = f" for {agent_name}" if agent_name else ""
                print(
                    f"Deleted {conv_count} conversations and {msg_count} messages{what}."
                )

        asyncio.run(main())

    except Exception as e:
        logger.exception("Failed to reset history")
        raise t.Exit(1) from e
