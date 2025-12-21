# /// script
# dependencies = ["agentpool"]
# ///

"""Example demonstrating team agent picking functionality."""

from __future__ import annotations

from agentpool import Agent, Team
from agentpool.docs.utils import run


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
