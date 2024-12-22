"""History management commands."""

from __future__ import annotations

from datetime import datetime

from llmling.cli.constants import output_format_opt
import typer as t

from llmling_agent.history import (
    StatsFilters,
    format_stats,
    get_conversation_stats,
    get_filtered_conversations,
    parse_time_period,
)
from llmling_agent.history.formatters import format_output


help_text = "Conversation history management"
history_cli = t.Typer(name="history", help=help_text, no_args_is_help=True)


@history_cli.command(name="show")
def show_history(
    agent_name: str | None = t.Argument(
        None,
        help="Agent name (shows all if not provided)",
    ),
    # Time-based filtering
    since: datetime | None = t.Option(  # noqa: B008
        None,
        "--since",
        "-s",
        help="Show conversations since (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    ),
    period: str | None = t.Option(
        None,
        "--period",
        "-p",
        help="Show conversations from last period (1h, 2d, 1w, 1m)",
    ),
    # Content filtering
    query: str | None = t.Option(
        None,
        "--query",
        "-q",
        help="Search in message content",
    ),
    model: str | None = t.Option(
        None,
        "--model",
        "-m",
        help="Filter by model used",
    ),
    # Output control
    limit: int = t.Option(10, "--limit", "-n", help="Number of conversations"),
    compact: bool = t.Option(
        False,
        "--compact",
        help="Show only first/last message of conversations",
    ),
    tokens: bool = t.Option(
        False,
        "--tokens",
        "-t",
        help="Include token usage statistics",
    ),
    output_format: str = output_format_opt,
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

        # Show GPT-4 conversations with token usage
        llmling-agent history show --model gpt-4 --tokens

        # Compact view of recent conversations
        llmling-agent history show --period 1d --compact
    """
    results = get_filtered_conversations(
        agent_name=agent_name,
        period=period,
        since=since,
        query=query,
        model=model,
        limit=limit,
        compact=compact,
        include_tokens=tokens,
    )
    print(format_output(results, output_format))  # type: ignore


@history_cli.command(name="stats")
def show_stats(
    agent_name: str | None = t.Argument(
        None,
        help="Agent name (shows all if not provided)",
    ),
    period: str = t.Option(
        "1d",
        "--period",
        "-p",
        help="Time period (1h, 1d, 1w, 1m, 1y)",
    ),
    group_by: str = t.Option(
        "agent",
        "--group-by",
        "-g",
        help="Group by: agent, model, hour, day",
    ),
    output_format: str = output_format_opt,
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
    cutoff = datetime.now() - parse_time_period(period)
    filters = StatsFilters(cutoff=cutoff, group_by=group_by, agent_name=agent_name)  # type: ignore
    stats = get_conversation_stats(filters)
    formatted = format_stats(stats, period, group_by)
    print(format_output(formatted, output_format))  # type: ignore


@history_cli.command(name="reset")
def reset_history(
    confirm: bool = t.Option(
        False,
        "--confirm",
        "-y",
        help="Confirm deletion without prompting",
    ),
    agent_name: str | None = t.Option(
        None,
        "--agent",
        "-a",
        help="Only delete history for specific agent",
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
    """
    from sqlalchemy import text
    from sqlmodel import Session, select

    from llmling_agent.storage import engine
    from llmling_agent.storage.models import Conversation, Message
    from llmling_agent.storage.queries import (
        DELETE_AGENT_CONVERSATIONS,
        DELETE_AGENT_MESSAGES,
        DELETE_ALL_CONVERSATIONS,
        DELETE_ALL_MESSAGES,
    )

    with Session(engine) as session:
        # Get count before deletion
        if agent_name:
            conv_query = select(Conversation).where(Conversation.agent_name == agent_name)
            msg_query = (
                select(Message)
                .join(Conversation)
                .where(Conversation.agent_name == agent_name)
            )
        else:
            conv_query = select(Conversation)
            msg_query = select(Message)

        conv_count = len(session.exec(conv_query).all())
        msg_count = len(session.exec(msg_query).all())

        if not conv_count:
            print("No conversations to delete.")
            return

        # Ask for confirmation if needed
        if not confirm:
            agent_str = f" for agent '{agent_name}'" if agent_name else ""
            msg = (
                f"This will delete {conv_count} conversations and {msg_count} messages"
                f"{agent_str}.\nAre you sure? [y/N] "
            )
            if input(msg).lower() != "y":
                print("Operation cancelled.")
                return

        # Delete messages first (foreign key constraint)
        if agent_name:
            # Delete messages from specific agent's conversations
            session.execute(
                text(DELETE_AGENT_MESSAGES),
                {"agent": agent_name},
            )
            # Delete conversations
            session.execute(
                text(DELETE_AGENT_CONVERSATIONS),
                {"agent": agent_name},
            )
        else:
            # Delete all
            session.execute(text(DELETE_ALL_MESSAGES))
            session.execute(text(DELETE_ALL_CONVERSATIONS))

        session.commit()

        agent_str = f" for {agent_name}" if agent_name else ""
        print(f"Deleted {conv_count} conversations and {msg_count} messages{agent_str}.")