"""Run command for agent execution."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import typer as t

from llmling_agent import AgentPool
from llmling_agent.log import get_logger
from llmling_agent_cli.cli_types import DetailLevel, LogLevel  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent import ChatMessage


logger = get_logger(__name__)


def watch_command(
    config: str = t.Argument(..., help="Path to agent configuration"),
    show_messages: bool = t.Option(
        True, "--show-messages", help="Show all messages (not just final responses)"
    ),
    detail_level: DetailLevel = t.Option(  # noqa: B008
        "simple", "-d", "--detail", help="Output detail level"
    ),
    show_metadata: bool = t.Option(False, "--metadata", help="Show message metadata"),
    show_costs: bool = t.Option(False, "--costs", help="Show token usage and costs"),
    log_level: LogLevel = t.Option("info", help="Logging level"),  # noqa: B008
):
    """Run agents in event-watching mode."""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    def on_message(chat_message: ChatMessage[Any]):
        text = chat_message.format(
            style=detail_level,  # type: ignore
            show_metadata=show_metadata,
            show_costs=show_costs,
        )
        print(text)

    async def run_watch():
        async with AgentPool(config) as pool:
            # Connect message handlers if showing all messages
            if show_messages:
                for agent in pool.agents.values():
                    agent.message_sent.connect(on_message)

            await pool.run_event_loop()

    asyncio.run(run_watch())
