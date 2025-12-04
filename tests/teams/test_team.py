"""Tests for Team execution."""

from __future__ import annotations

import pytest

from llmling_agent import Agent, AgentPool, ChatMessage, Team, TeamRun


async def test_team_parallel_execution():
    """Test that team runs all agents in parallel and collects responses."""
    async with AgentPool() as pool:
        # Create three agents that append their name to input
        a1 = await pool.add_agent("a1", system_prompt="Append 'a1'", model="test")
        a2 = await pool.add_agent("a2", system_prompt="Append 'a2'", model="test")
        a3 = await pool.add_agent("a3", system_prompt="Append 'a3'", model="test")

        team = pool.create_team([a1, a2, a3])
        result = await team.execute("test")

        # Check that we got responses from all agents
        assert len(result) == 3  # noqa: PLR2004
        agent_names = {r.agent_name for r in result}
        assert agent_names == {"a1", "a2", "a3"}

        # Check that stats were collected
        assert len(team.execution_stats.messages) == 3  # noqa: PLR2004
        assert all(isinstance(msg, ChatMessage) for msg in team.execution_stats.messages)


async def test_team_shared_prompt():
    """Test that shared prompt is prepended to individual prompts."""
    async with AgentPool() as pool:
        # Create agents that echo their input
        def echo(prompt: str) -> str:
            return prompt

        a1 = pool.get_agent(Agent.from_callback(echo, name="a1"))
        a2 = pool.get_agent(Agent.from_callback(echo, name="a2"))

        # Create team with shared prompt
        team = pool.create_team([a1, a2], shared_prompt="Common instruction: ")
        result = await team.execute("specific task")

        # Each agent should get both prompts
        assert len(result) == 2  # noqa: PLR2004
        for response in result:
            assert response.message
            assert "Common instruction" in str(response.message.content)
            assert "specific task" in str(response.message.content)


async def test_nested_teams():
    """Test nesting Teams and TeamRuns inside each other."""
    async with AgentPool() as pool:
        # Create basic agents
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")
        a3 = await pool.add_agent("a3", model="test")

        # Case 1: Team inside TeamRun
        team = a1 & a2  # Team of two agents
        execution = team | a3  # TeamRun with Team + Agent
        result = await execution.run("test message")
        assert isinstance(result, ChatMessage)
        # Team's messages should be in the chain
        assert (
            len(execution.execution_stats.messages) == 2  # noqa: PLR2004
        )  # Team(a1+a2) + a3


async def test_nested_team_run():
    """Test nesting Teams and TeamRuns inside each other."""
    async with AgentPool() as pool:
        # Create basic agents
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")
        a3 = await pool.add_agent("a3", model="test")
        a4 = await pool.add_agent("a4", model="test")

        # Case 2: TeamRun inside Team
        sequential = a1 | a2  # TeamRun
        parallel_team = Team([sequential, a3, a4])  # Team containing TeamRun + Agents

        result = await parallel_team.run("test message")
        assert isinstance(result, ChatMessage)
        # Should have all messages
        assert (
            len(parallel_team.execution_stats.messages) == 3  # noqa: PLR2004
        )  # TeamRun(a1+a2) + a3 + a4

        # # Test streaming with nested Team
        # async with execution.run_stream("test message") as stream:
        #     chunks = [chunk async for chunk in stream.stream_output()]
        #     assert chunks  # Should get chunks from all agents

        # Test iteration with nested TeamRun
        messages = [msg async for msg in parallel_team.run_iter("test message")]
        assert len(messages) == 3  # Should get all messages  # noqa: PLR2004


async def test_simple_team_run_iter():
    """Test run_iter with a simple team of agents."""
    async with AgentPool() as pool:
        # Create basic agents
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")

        # Simple parallel team
        team = pool.create_team([a1, a2])

        # Test iteration
        messages = [msg async for msg in team.run_iter("test message")]
        assert len(messages) == 2  # Should get one message per agent  # noqa: PLR2004
        assert {msg.name for msg in messages} == {"a1", "a2"}


async def test_sequential_run_iter():
    """Test run_iter with a sequential execution (TeamRun)."""
    async with AgentPool() as pool:
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")

        # Sequential execution
        sequential = a1 | a2

        messages = [msg async for msg in sequential.run_iter("test message")]
        assert len(messages) == 2  # noqa: PLR2004
        # Should maintain order
        assert [msg.name for msg in messages] == ["a1", "a2"]


async def test_simple_team_with_teamrun_iter():
    """Test run_iter with a team containing a simple TeamRun."""
    async with AgentPool() as pool:
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")
        a3 = await pool.add_agent("a3", model="test")

        # Sequential execution as team member
        sequential = a1 | a2  # This is one unit
        # Team with two members: sequential and a3
        team = pool.create_team([sequential, a3])

        messages = [msg async for msg in team.run_iter("test message")]

        # Should get TWO messages: one from TeamRun, one from a3
        assert len(messages) == 2  # noqa: PLR2004

        # Verify senders
        senders = {msg.name for msg in messages}
        assert senders == {sequential.name, "a3"}

        # Verify TeamRun message has metadata about its internal execution
        teamrun_msg = next(msg for msg in messages if msg.name == sequential.name)
        assert "execution_order" in teamrun_msg.metadata


async def test_team_run_iter_execution_order():
    """Test that run_iter preserves execution order within sequential parts."""
    async with AgentPool() as pool:
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")
        a3 = await pool.add_agent("a3", model="test")

        # Sequential execution
        sequential: TeamRun[None, str] = pool.create_team_run([a1, a2], name="sequential")
        # Team with sequential + single agent
        team = pool.create_team([sequential, a3], name="parallel")

        messages = [msg async for msg in team.run_iter("test message")]
        # Find sequential messages
        seq_msgs = [msg for msg in messages if msg.name == sequential.name]
        seq_msg = seq_msgs[0]
        # Order should be preserved within sequential execution
        assert [msg.name for msg in seq_msg.associated_messages] == ["a1", "a2"]


async def test_team_operators():
    """Test team combination operators (& and |)."""
    async with AgentPool() as pool:
        # Create basic agents
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")
        a3 = await pool.add_agent("a3", model="test")
        a4 = await pool.add_agent("a4", model="test")

        # Test parallel combinations (&)
        # Simple agent combinations
        team1 = a1 & a2
        assert isinstance(team1, Team)
        assert len(team1.nodes) == 2  # noqa: PLR2004
        assert list(team1.nodes) == [a1, a2]

        # Adding agent to team
        team2 = team1 & a3
        assert isinstance(team2, Team)
        assert len(team2.nodes) == 3  # noqa: PLR2004
        assert list(team2.nodes) == [a1, a2, a3]

        # Combining teams - should flatten
        other_team = a3 & a4
        combined = team1 & other_team
        assert isinstance(combined, Team)
        assert len(combined.nodes) == 4  # noqa: PLR2004
        assert list(combined.nodes) == [a1, a2, a3, a4]

        # Test sequential combinations (|)
        # Simple agent combinations
        seq1 = a1 | a2
        assert isinstance(seq1, TeamRun)
        assert len(seq1.nodes) == 2  # noqa: PLR2004
        assert list(seq1.nodes) == [a1, a2]

        # Adding to TeamRun - should extend
        seq2 = seq1 | a3
        assert seq2 is seq1  # Same TeamRun instance
        assert len(seq1.nodes) == 3  # noqa: PLR2004
        assert list(seq1.nodes) == [a1, a2, a3]

        # Complex combinations
        team3 = a1 & a2  # parallel
        seq3 = a3 | a4  # sequential

        # TeamRun with Team member
        combined_1 = team3 | seq3
        assert isinstance(combined_1, TeamRun)
        assert len(combined_1.nodes) == 2  # noqa: PLR2004
        assert combined_1.nodes[0] is team3
        assert isinstance(combined_1.nodes[1], TeamRun)

        # Team with TeamRun member
        combined_2 = Team([team3, seq3])
        assert isinstance(combined_2, Team)
        assert len(combined_2.nodes) == 2  # noqa: PLR2004
        assert combined_2.nodes[0] is team3
        assert isinstance(combined_2.nodes[1], TeamRun)

        # Test actual execution
        result = await combined_1.run("test")
        assert isinstance(result, ChatMessage)
        # All nodes should have executed
        assert len(combined_1.execution_stats.messages) == len(combined_1.nodes)

        result = await combined_2.run("test")
        assert isinstance(result, ChatMessage)
        # All nodes should have executed
        assert len(combined_2.execution_stats.messages) == len(combined_2.nodes)


if __name__ == "__main__":
    pytest.main([__file__])
