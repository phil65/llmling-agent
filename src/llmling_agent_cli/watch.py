"""Run command for agent execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import typer as t

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage


logger = get_logger(__name__)


def watch_command(
    config: str = t.Argument(..., help="Path to agent configuration"),
    show_messages: bool = t.Option(
        True, "--show-messages", help="Show all messages (not just final responses)"
    ),
    detail_level: str = t.Option(
        "simple", "-d", "--detail", help="Output detail level: simple/detailed/markdown"
    ),
    show_metadata: bool = t.Option(False, "--metadata", help="Show message metadata"),
    show_costs: bool = t.Option(False, "--costs", help="Show token usage and costs"),
    log_level: str = t.Option("INFO", help="Logging level"),
):
    """Run agents in event-watching mode."""

    async def run_watch():
        async with AgentPool[None](config) as pool:

            def on_message(chat_message: ChatMessage[Any]):
                print(
                    chat_message.format(
                        style=detail_level,  # type: ignore
                        show_metadata=show_metadata,
                        show_costs=show_costs,
                    )
                )

            # Connect message handlers if showing all messages
            if show_messages:
                for agent in pool.agents.values():
                    agent.message_sent.connect(on_message)

            await pool.run_event_loop()

    asyncio.run(run_watch())
