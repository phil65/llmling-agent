from __future__ import annotations

import pytest

from llmling_agent import Agent


async def test_pick_from_options(default_model: str):
    """Test picking from a list of options."""
    # Create agent for making decisions
    decider = Agent(
        model=default_model,
        system_prompt="You are an expert at making clear decisions.",
    )

    # Test picking from simple options
    options = ["A", "B", "C"]

    decision = await decider.talk.pick(options, task="Pick A and give a random reason!")

    assert decision.selection in options
    assert len(decision.reason) > 0


@pytest.mark.flaky(reruns=2)
async def test_pick_from_agents(default_model: str):
    """Test picking from a team of agents."""
    # Create a team of specialized agents
    analyzer = Agent(
        name="code_analyzer",
        model=default_model,
        description="Specializes in code analysis and best practices",
    )
    reviewer = Agent(
        name="security_expert",
        model=default_model,
        description="Focuses on security vulnerabilities",
    )
    team = [analyzer, reviewer]

    # Create decision maker
    decider = Agent(
        model=default_model,
        system_prompt="You are an expert at delegating tasks.",
    )

    # Test agent selection
    decision = await decider.talk.pick(
        team,
        task="We found potential SQL injection vulnerabilities. Who should investigate?",
    )

    assert decision.selection in team
    assert decision.selection.name == "security_expert"  # Should pick security expert
    assert "security" in decision.reason.lower()


async def test_pick_multiple(default_model: str):
    """Test picking multiple options with constraints."""
    decider = Agent(model=default_model)

    decision = await decider.talk.pick_multiple(
        ["A", "B", "C"],
        task="Pick A and B. Always pick both!.",
        min_picks=2,
        max_picks=2,
    )

    assert decision.selections == ["A", "B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
