from __future__ import annotations

import pytest

from llmling_agent import AgentPool


async def test_simple_sequential_chain():
    """Test basic sequential chaining."""
    async with AgentPool() as pool:
        agent1 = await pool.add_agent("agent1", model="test")
        agent2 = await pool.add_agent("agent2", model="test")
        agent3 = await pool.add_agent("agent3", model="test")
        agent1 >> agent2 >> agent3
        async with pool.track_message_flow() as tracker:
            msg = await agent1.run("test")
            mermaid = tracker.visualize(msg)
            # Should only see these two connections
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == sorted(["agent1-->agent2", "agent2-->agent3"])


async def test_parallel_to_sequential():
    """Test parallel flows connecting to single target."""
    async with AgentPool() as pool:
        agent1 = await pool.add_agent("agent1", model="test")
        agent2 = await pool.add_agent("agent2", model="test")
        agent3 = await pool.add_agent("agent3", model="test")
        agent4 = await pool.add_agent("agent4", model="test")
        agent1 >> [agent2, agent3] >> agent4
        async with pool.track_message_flow() as tracker:
            msg = await agent1.run("test")
            mermaid = tracker.visualize(msg)
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == sorted([
                "agent1-->agent2",
                "agent1-->agent3",
                "agent2-->agent4",
                "agent3-->agent4",
            ])


async def test_callback_chain():
    """Test chaining with a callback function."""
    async with AgentPool() as pool:
        agent1 = await pool.add_agent("agent1", model="test")
        agent2 = await pool.add_agent("agent2", model="test")

        def process(msg: str) -> str:
            return f"Processed: {msg}"

        _talk = agent1 >> process >> agent2
        async with pool.track_message_flow() as tracker:
            msg = await agent1.run("test")
            mermaid = tracker.visualize(msg)
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == sorted(["agent1-->process", "process-->agent2"])


async def test_message_flow_tracker():
    """Test tracking and visualizing message flow through a chain."""
    # Setup a simple agent chain
    async with AgentPool() as pool:
        agent1 = await pool.add_agent(
            "agent1",
            system_prompt="You are agent 1",
            model="test",
        )
        agent2 = await pool.add_agent(
            "agent2",
            system_prompt="You are agent 2",
            model="test",
        )
        agent3 = await pool.add_agent(
            "agent3",
            system_prompt="You are agent 3",
            model="test",
        )

        # Create chain: agent1 >> agent2 >> agent3
        agent1 >> agent2
        agent2 >> agent3

        # Track message flow during execution
        async with pool.track_message_flow() as tracker:
            result = await agent1.run("Hello")

            # Get flow visualization
            mermaid = tracker.visualize(result)

            # Check for expected connections in diagram
            assert "flowchart LR" in mermaid
            assert "agent1-->agent2" in mermaid.replace(" ", "")
            assert "agent2-->agent3" in mermaid.replace(" ", "")

            # Should not contain non-existent connections
            assert "agent1-->agent3" not in mermaid.replace(" ", "")
            assert "agent3-->agent1" not in mermaid.replace(" ", "")

        # Tracker should no longer receive events after context exit
        assert len(tracker.events) > 0  # Should have events from the run
        previous_count = len(tracker.events)

        # Run again outside context
        await agent1.run("Another message")
        assert len(tracker.events) == previous_count  # No new events tracked


async def test_message_flow_tracker_parallel():
    """Test tracking parallel message flows."""
    async with AgentPool() as pool:
        agent1 = await pool.add_agent("agent1", model="test")
        agent2 = await pool.add_agent("agent2", model="test")
        agent3 = await pool.add_agent("agent3", model="test")

        # Create parallel flows: agent1 >> [agent2, agent3]
        agent1 >> [agent2, agent3]

        async with pool.track_message_flow() as tracker:
            result = await agent1.run("Hello")
            mermaid = tracker.visualize(result)

            # Both parallel paths should be in diagram
            assert "agent1-->agent2" in mermaid.replace(" ", "")
            assert "agent1-->agent3" in mermaid.replace(" ", "")

            # Only show flows for this conversation
            other_result = await agent1.run("Different conversation")
            other_mermaid = tracker.visualize(other_result)

            # Each visualization should only show its own conversation
            assert mermaid == other_mermaid


async def test_message_flow_tracker_nested():
    """Test tracking flow through nested teams."""
    async with AgentPool() as pool:
        agent1 = await pool.add_agent("agent1", model="test")
        agent2 = await pool.add_agent("agent2", model="test")
        agent3 = await pool.add_agent("agent3", model="test")

        # Create nested team
        team = pool.create_team([agent2, agent3], name="team")
        agent1 >> team

        async with pool.track_message_flow() as tracker:
            result = await agent1.run("Hello")
            mermaid = tracker.visualize(result)

            # Should only show connection to team as a unit
            connections = mermaid.replace(" ", "").split("\n")[1:]  # pyright: ignore
            assert sorted(connections) == ["agent1-->team"]


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
