from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import AgentPool


@pytest.fixture
async def pool():
    """Create agent pool with test agents."""
    pool = AgentPool()

    # Add three agents with test models
    await pool.add_agent("agent1", model=TestModel())
    await pool.add_agent("agent2", model=TestModel())
    await pool.add_agent("agent3", model=TestModel())

    async with pool:
        yield pool


async def test_registry_captures_agent_interaction(pool: AgentPool):
    """Test that registry captures real agent interactions."""
    messages = []
    pool.connection_registry.message_flow.connect(lambda event: messages.append(event))

    # Get agents and set up connection
    agent1 = pool.get_agent("agent1")
    agent2 = pool.get_agent("agent2")
    agent1.connect_to(agent2, name="test_talk")

    # Trigger actual interaction
    await agent1.run("Test message")

    # Verify flow was captured
    assert len(messages) == 1
    assert messages[0].source == agent1
    assert messages[0].targets == [agent2]


async def test_chained_communication(pool: AgentPool):
    """Test message flow through chain of agents."""
    messages = []
    pool.connection_registry.message_flow.connect(lambda event: messages.append(event))

    # Set up chain: agent1 -> agent2 -> agent3
    agent1 = pool.get_agent("agent1")
    agent2 = pool.get_agent("agent2")
    agent3 = pool.get_agent("agent3")

    # Create chain with named connections
    agent1.connect_to(agent2, name="chain1")
    agent2.connect_to(agent3, name="chain2")

    # Trigger chain
    await agent1.run("Start chain")

    # Should capture both flows
    assert len(messages) == 2  # noqa: PLR2004
    assert messages[0].source == agent1
    assert messages[0].targets == [agent2]
    assert messages[1].source == agent2
    assert messages[1].targets == [agent3]


async def test_broadcast_communication(pool: AgentPool):
    """Test broadcasting to multiple agents."""
    messages = []
    pool.connection_registry.message_flow.connect(lambda event: messages.append(event))

    # Set up broadcast: agent1 -> [agent2, agent3]
    agent1 = pool.get_agent("agent1")
    agent2 = pool.get_agent("agent2")
    agent3 = pool.get_agent("agent3")

    # Create individual connections for broadcast
    agent1.connect_to(agent2, name="broadcast1")
    agent1.connect_to(agent3, name="broadcast2")

    # Send broadcast
    await agent1.run("Broadcast message")

    # Should capture two events, one for each target
    assert len(messages) == 2  # noqa: PLR2004
    targets = {t for m in messages for t in m.targets}
    assert targets == {agent2, agent3}
    assert all(m.source == agent1 for m in messages)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
