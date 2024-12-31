"""Web interface commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
from tempfile import NamedTemporaryFile

from slashed import DefaultOutputWriter
import typer as t

from llmling_agent.delegation import AgentPool
from llmling_agent.log import set_handler_level
from llmling_agent_cli.chat_session.session import start_interactive_session


THEME_HELP = "UI theme (soft/base/monochrome/glass/default)"
MODEL_HELP = "Model to use (e.g. openai:gpt-4o-mini, gpt-4)"

QUICKSTART_CONFIG = """\
agents:
    quickstart:
        name: quickstart
        model: {model}
"""


def quickstart_command(
    model: str = t.Argument("openai:gpt-4o-mini", help=MODEL_HELP),
    log_level: str = t.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
    stream: bool = t.Option(
        True,
        "--stream/--no-stream",
        help="Enable streaming mode (default: off)",
    ),
):
    """Start an ephemeral chat session with minimal setup."""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    try:
        # Create temporary agent config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_agent:
            tmp_agent.write(QUICKSTART_CONFIG.format(model=model))
            agent_path = tmp_agent.name

        async def run_chat():
            async with AgentPool.open(agent_path, agents=["quickstart"]) as pool:
                agent = pool.get_agent("quickstart")
                await start_interactive_session(agent, pool=pool, stream=stream)

        show_logs = False
        output = DefaultOutputWriter() if show_logs else None
        loggers = ["llmling_agent", "llmling"]
        with set_handler_level(level, loggers, session_handler=output):
            asyncio.run(run_chat())

    except KeyboardInterrupt:
        print("\nChat session ended.")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        raise t.Exit(1) from None
    finally:
        # Cleanup temporary file
        with contextlib.suppress(Exception):
            pathlib.Path(agent_path).unlink(missing_ok=True)
