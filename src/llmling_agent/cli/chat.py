"""Interactive chat command."""

from __future__ import annotations

import asyncio
import logging

from llmling.core.log import get_logger
import typer as t

from llmling_agent import LLMlingAgent
from llmling_agent.cli import resolve_agent_config


logger = get_logger(__name__)


def chat_command(
    agent_name: str = t.Argument(help="Name of agent to chat with"),
    config: str | None = t.Option(
        None, "--config", "-c", help="Override agent config path"
    ),
    model: str | None = t.Option(None, "--model", "-m", help="Override agent's model"),
    log_level: str = t.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Start interactive chat session with an agent."""
    from llmling_agent.cli.chat_session.session import start_interactive_session

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)
    logging.getLogger("llmling_agent").setLevel(level)
    logging.getLogger("llmling").setLevel(level)

    try:
        # Resolve configuration
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        async def run_chat() -> None:
            async with LLMlingAgent[str].open_agent(
                config_path,
                agent_name,
                model=model,  # type: ignore[arg-type]
            ) as agent:
                await start_interactive_session(agent, log_level=level)

        asyncio.run(run_chat())

    except t.Exit:
        raise
    except KeyboardInterrupt:
        print("\nChat session ended.")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        raise t.Exit(1)  # noqa: B904


if __name__ == "__main__":
    chat_command()
