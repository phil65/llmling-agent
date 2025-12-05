"""Run command for agent execution."""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING, Any

import typer as t

from llmling_agent import AgentPool
from llmling_agent_cli import resolve_agent_config
from llmling_agent_cli.cli_types import DetailLevel  # noqa: TC001
from llmling_agent_cli.common import verbose_opt


if TYPE_CHECKING:
    from llmling_agent import ChatMessage


def run_command(
    node_name: str = t.Argument(help="Agent / Team name to run"),
    prompts: list[str] = t.Argument(None, help="Additional prompts to send"),  # noqa: B008
    config_path: str = t.Option(None, "-c", "--config", help="Override config path"),
    show_messages: bool = t.Option(
        True, "--show-messages", help="Show all messages (not just final responses)"
    ),
    detail_level: DetailLevel = t.Option(  # noqa: B008
        "simple", "-d", "--detail", help="Output detail level"
    ),
    show_metadata: bool = t.Option(False, "--metadata", help="Show message metadata"),
    show_costs: bool = t.Option(False, "--costs", help="Show token usage and costs"),
    verbose: bool = verbose_opt,
) -> None:
    """Single-shot run a node (agent/team) with prompts."""
    try:
        # Resolve configuration path
        try:
            config_path = resolve_agent_config(config_path)
        except ValueError as e:
            error_msg = str(e)
            raise t.BadParameter(error_msg) from e

        async def run() -> None:
            async with AgentPool(config_path) as pool:

                def on_message(chat_message: ChatMessage[Any]) -> None:
                    print(
                        chat_message.format(
                            style=detail_level,
                            show_metadata=show_metadata,
                            show_costs=show_costs,
                        )
                    )

                # Connect message handlers if showing all messages
                if show_messages:
                    for node in pool.nodes.values():
                        node.message_sent.connect(on_message)
                for prompt in prompts:
                    response = await pool.nodes[node_name].run(prompt)

                    if not show_messages:
                        print(
                            response.format(
                                style=detail_level,
                                show_metadata=show_metadata,
                                show_costs=show_costs,
                            )
                        )

        # Run the async code in the sync command
        asyncio.run(run())

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
