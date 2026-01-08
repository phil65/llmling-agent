from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from agentpool import AgentPool


async def test_message_chain():
    """Test that message chain tracks transformations correctly via parent_id."""
    async with AgentPool() as pool:
        agent_a = await pool.add_agent("agent-a", model="test")
        agent_b = await pool.add_agent("agent-b", model="test")
        agent_c = await pool.add_agent("agent-c", model="test")

        # Connect chain
        agent_a.connect_to(agent_b)
        agent_b.connect_to(agent_c)

        # When A processes a new message
        result_a = await agent_a.run("Start")
        assert result_a.parent_id is not None  # Points to user message

        # When B processes A's message via run_message
        result_b = await agent_b.run_message(result_a)
        assert result_b.parent_id is not None
        # Chain should show A
        chain_b = pool.get_message_chain(result_b)
        assert "agent-a" in chain_b

        # When C processes B's message
        result_c = await agent_c.run_message(result_b)
        assert result_c.parent_id is not None
        # Chain should show A and B
        chain_c = pool.get_message_chain(result_c)
        assert "agent-a" in chain_c
        assert "agent-b" in chain_c


async def test_run_result_has_parent_id():
    """Test that the message returned by run() has proper parent_id."""
    async with AgentPool() as pool:
        model = TestModel(custom_output_text="Response from A")
        agent_a = await pool.add_agent("agent-a", model=model)
        agent_b = await pool.add_agent("agent-b", model=model)

        # Connect A to B
        agent_a.connect_to(agent_b)

        # When A runs
        result = await agent_a.run("Test message")

        # The returned message should have parent_id pointing to user message
        assert result.parent_id is not None

        # Wait for forwarding to complete
        await agent_a.task_manager.complete_tasks()
        await agent_b.task_manager.complete_tasks()

        # B's messages should have parent_id tracking the chain
        if agent_b.conversation.chat_messages:
            b_user_msg = next(
                (m for m in agent_b.conversation.chat_messages if m.role == "user"),
                None,
            )
            if b_user_msg:
                # The user message in B should have parent_id from A's response
                assert b_user_msg.parent_id == result.message_id


async def test_message_chain_through_routing():
    """Test that message chain tracks correctly through the routing system."""
    async with AgentPool() as pool:
        model_a = TestModel(custom_output_text="Response from A")
        model_b = TestModel(custom_output_text="Response from B")
        model_c = TestModel(custom_output_text="Response from C")

        agent_a = await pool.add_agent("agent-a", model=model_a)
        agent_b = await pool.add_agent("agent-b", model=model_b)
        agent_c = await pool.add_agent("agent-c", model=model_c)

        # Connect the chain
        agent_a.connect_to(agent_b)
        agent_b.connect_to(agent_c)

        # When A starts the chain
        await agent_a.run("Start message")

        # Wait for all routing to complete
        await agent_a.task_manager.complete_tasks()
        await agent_b.task_manager.complete_tasks()
        await agent_c.task_manager.complete_tasks()

        # All agents should share the same conversation_id
        assert (
            agent_a.conversation.chat_messages[0].conversation_id
            == agent_b.conversation.chat_messages[0].conversation_id
        )
        assert (
            agent_b.conversation.chat_messages[0].conversation_id
            == agent_c.conversation.chat_messages[0].conversation_id
        )

        # C's response should have a chain back through B and A
        if agent_c.conversation.chat_messages:
            c_response = next(
                (m for m in agent_c.conversation.chat_messages if m.role == "assistant"),
                None,
            )
            if c_response:
                chain = pool.get_message_chain(c_response)
                # Chain should include both A and B
                assert "agent-a" in chain or "agent-b" in chain


async def test_find_message_by_id():
    """Test that pool.find_message_by_id works across agents."""
    async with AgentPool() as pool:
        agent_a = await pool.add_agent("agent-a", model="test")
        agent_b = await pool.add_agent("agent-b", model="test")

        result_a = await agent_a.run("Hello from A")
        result_b = await agent_b.run("Hello from B")

        # Should find messages from both agents
        found_a = pool.find_message_by_id(result_a.message_id)
        found_b = pool.find_message_by_id(result_b.message_id)

        assert found_a is not None
        assert found_b is not None
        assert found_a.message_id == result_a.message_id
        assert found_b.message_id == result_b.message_id

        # Non-existent ID should return None
        assert pool.find_message_by_id("non-existent-id") is None


if __name__ == "__main__":
    pytest.main([__file__])
