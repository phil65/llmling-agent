---
title: Expert Selection
description: Using pick() for expert selection
icon: material/account-group
---


## Try it in Pydantic Playground




## Files

- `config.yml`
- `main.py`
- `pick_teams.py`



# Expert Selection with pick() and pick_multiple()


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
agents:
  coordinator:
    model: openrouter:openai/gpt-5-nano
    system_prompts:
      - You select the most suitable expert(s) for each task.

  database_expert:
    model: openrouter:openai/gpt-5-nano
    description: Expert in SQL optimization and database design.

  frontend_dev:
    model: openrouter:openai/gpt-5-nano
    description: Specialist in React and modern web interfaces.

  security_expert:
    model: openrouter:openai/gpt-5-nano
    description: Expert in penetration testing and security audits.

```


### `main.py`

```py
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


async def run_example() -> None:
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

```


### `pick_teams.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///

"""Example demonstrating team agent picking functionality."""

from __future__ import annotations

from llmling_agent import Agent, Team
from llmling_agent_docs.examples.utils import run


async def main() -> None:
    # Parallel team members
    developer = Agent(
        name="developer",
        description="Implements new code features and changes",
        model="gpt-5-mini",
        system_prompt="You write Python code and implement features.",
    )

    doc_writer = Agent(
        name="doc_writer",
        description="Writes and updates technical documentation",
        model="gpt-5-mini",
        system_prompt="You specialize in writing technical documentation.",
    )

    lazy_bob = Agent(
        name="lazy_bob",
        description="Has no useful skills or contributions",
        model="gpt-5-mini",
        system_prompt="You avoid work at all costs.",
    )

    team_lead = Agent(
        name="team_lead",
        model="gpt-5-mini",
        system_prompt="You assign work to team members based on their skills.",
    )
    feature_team = Team([developer, doc_writer, lazy_bob], picker=team_lead)
    print("\n=== Parallel Team Example ===")
    task = "Implement a new sort_by_date() function and document it in the API guide."
    async for msg in feature_team.run_iter(task):
        print(f"{msg.name}: {msg.content}")


if __name__ == "__main__":
    run(main())

```



This example demonstrates LLMling-agent's type-safe selection methods:

- Using pick() for single expert selection
- Using pick_multiple() for team selection
- Type-safe results with reasoning
- Team-based agent selection


## How It Works

1. Single Selection (pick):
   - Takes a team of agents and a task description
   - Returns a single expert with reasoning
   - Result is type-safe: `Pick[Agent]`

2. Multiple Selection (pick_multiple):
   - Takes same inputs plus min/max picks
   - Returns multiple experts with reasoning
   - Result is type-safe: `MultiPick[Agent]`

Example Output:
```
Selected: database_expert
Reason: The task specifically involves SQL query optimization, which is the database expert's primary specialty.

Selected: frontend_dev, security_expert
Reason: Creating a secure login page requires both frontend expertise for the user interface and security expertise for proper authentication implementation.
```

This demonstrates:

- Type-safe agent selection
- Reasoned decision-making
- Team-based operations
- Flexible expert allocation

