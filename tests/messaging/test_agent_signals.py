from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from agentpool import Agent


if TYPE_CHECKING:
    from agentpool import ChatMessage


async def test_message_chain():
    """Test that message chain tracks transformations correctly."""
    async with Agent(name="agent-a", model="test") as agent_a:  # noqa: SIM117
        async with Agent(name="agent-b", model="test") as agent_b:
            async with Agent(name="agent-c", model="test") as agent_c:
                # Connect chain
                agent_a.connect_to(agent_b)
                agent_b.connect_to(agent_c)

                # When A processes a new message
                result_a = await agent_a.run("Start")
                assert result_a.forwarded_from == []  # New message, empty chain

                # When B processes A's message
                result_b = await agent_b.run(result_a)
                assert result_b.forwarded_from == [agent_a.name]  # Chain includes A

                # When C processes B's message
                result_c = await agent_c.run(result_b)
                assert result_c.forwarded_from == [agent_a.name, agent_b.name]


async def test_run_result_not_modified_by_connections():
    """Test that the message returned by run() isn't modified by connections."""
    # Create two agents
    model = TestModel(custom_output_text="Response from B")
    async with Agent(name="agent-a", model=model) as agent_a:  # noqa: SIM117
        async with Agent(name="agent-b", model=model) as agent_b:
            # Connect A to B
            agent_a.connect_to(agent_b)

            # When A runs
            result = await agent_a.run("Test message")

            # Then the returned message should only contain A as source
            assert result.forwarded_from == [], "run() result should have empty forwarded_from"
            # or possibly just [agent_a.name] if we decide that's the expected behavior

            # While messages received by B should have the full chain
            def collect_b(msg: ChatMessage[Any]):
                assert msg.forwarded_from == ["agent-a"], "Forwarded message should contain chain"

            agent_b.message_sent.connect(collect_b)


async def test_message_chain_through_routing():
    """Test that message chain tracks correctly through the routing system."""
    model_b = TestModel(custom_output_text="Response from B")
    model_c = TestModel(custom_output_text="Response from C")

    async with Agent(name="agent-a", model="test") as agent_a:  # noqa: SIM117
        async with Agent(name="agent-b", model=model_b) as agent_b:
            async with Agent(name="agent-c", model=model_c) as agent_c:
                # Track messages received by C
                received_by_c: list[ChatMessage[Any]] = []
                agent_c.message_received.connect(received_by_c.append)

                # Connect the chain
                agent_a.connect_to(agent_b)
                agent_b.connect_to(agent_c)

                # When A starts the chain
                await agent_a.run("Start message")

                # Then C should receive message with complete chain
                assert len(received_by_c) == 1
                final_msg = received_by_c[0]
                assert final_msg.forwarded_from == ["agent-a", "agent-b"]
                assert "Response from B" in final_msg.content
                assert agent_a.conversation.chat_messages[0].forwarded_from == []
                assert agent_b.conversation.chat_messages[0].forwarded_from == ["agent-a"]
                assert agent_b.conversation.chat_messages[1].forwarded_from == [
                    "agent-a"  # TODO: how to handle this? Should both messages have this?
                ]
                assert agent_c.conversation.chat_messages[0].forwarded_from == [
                    "agent-a",
                    "agent-b",
                ]
                assert (
                    agent_a.conversation.chat_messages[0].conversation_id
                    == agent_b.conversation.chat_messages[0].conversation_id
                )

                assert (
                    agent_b.conversation.chat_messages[0].conversation_id
                    == agent_c.conversation.chat_messages[0].conversation_id
                )


if __name__ == "__main__":
    pytest.main([__file__])
