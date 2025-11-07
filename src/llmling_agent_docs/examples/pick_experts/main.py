# /// script
# dependencies = ["llmling-agent"]
# ///

"""Example: Using pick() and pick_multiple() for expert selection."""

from __future__ import annotations

import os

from llmling_agent import AgentPool
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


async def run_example():
    """Run the expert selection example."""
    config_path = get_config_path(None if is_pyodide() else __file__)
    async with AgentPool(config_path) as pool:
        coordinator = pool.get_agent("coordinator")
        experts = pool.create_team(["database_expert", "frontend_dev", "security_expert"])

        # Single expert selection
        task = "Who should optimize our slow-running SQL queries?"
        pick = await coordinator.talk.pick(experts, task=task)
        # the result is type safe, pick.selection is an agent instance
        assert pick.selection in experts
        print(f"Selected: {pick.selection.name} Reason: {pick.reason}")

        # Multiple expert selection
        task = "Who should we assign to create a secure login page?"
        multi_pick = await coordinator.talk.pick_multiple(experts, task=task, min_picks=2)
        # also here type-safe result
        selected = ", ".join(e.name for e in multi_pick.selections)
        print(f"Selected: {selected} Reason: {multi_pick.reason}")


if __name__ == "__main__":
    run(run_example())
