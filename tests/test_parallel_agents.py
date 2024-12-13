"""Tests for parallel agent execution."""

from __future__ import annotations

import pytest

from llmling_agent.models import (
    AgentConfig,
    AgentDefinition,
    ResponseDefinition,
    ResponseField,
)
from llmling_agent.runners import AgentOrchestrator, AgentRunConfig


@pytest.mark.asyncio
async def test_parallel_agent_execution(test_model):
    """Test multiple agents executing the same prompts in parallel.

    The orchestrator runs each prompt through all agents, allowing comparison
    of how different agents handle the same input.
    """
    agent_def = AgentDefinition(
        responses={
            "BasicResult": ResponseDefinition(
                description="Basic test result",
                fields={"message": ResponseField(type="str", description="Test message")},
            )
        },
        agents={
            "agent1": AgentConfig(
                name="First Agent",
                model=test_model,
                result_type="BasicResult",
                system_prompts=["You are the first agent"],
            ),
            "agent2": AgentConfig(
                name="Second Agent",
                model=test_model,
                result_type="BasicResult",
                system_prompts=["You are the second agent"],
            ),
        },
    )

    # Run same prompt through multiple agents
    config = AgentRunConfig(
        agent_names=["agent1", "agent2"],
        prompts=["Process this input"],  # Same prompt for all agents
    )

    orchestrator = AgentOrchestrator(agent_def, config)
    results = await orchestrator.run()

    # Verify each agent processed the prompt
    assert "agent1" in results
    assert "agent2" in results
    assert len(results["agent1"]) == 1
    assert len(results["agent2"]) == 1

    # Both agents should return test model's response
    for agent_results in results.values():
        assert len(agent_results) == 1
        result = agent_results[0]
        assert result.data == "Test response"