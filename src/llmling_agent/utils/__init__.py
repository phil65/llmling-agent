"""Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import jinja2


def setup_env(env: jinja2.Environment):
    """Used as extension point for the jinjarope environment.

    Args:
        env: The jinjarope environment to extend
    """
    from llmling_agent.agent.agent import Agent
    from llmling_agent.functional import (
        run_agent,
        run_agent_sync,
        get_structured,
        get_structured_multiple,
        pick_one,
    )

    env.globals |= dict(agent=Agent)
    env.filters |= {
        "run_agent": run_agent,
        "run_agent_sync": run_agent_sync,
        "pick_one": pick_one,
        "get_structured": get_structured,
        "get_structured_multiple": get_structured_multiple,
    }
