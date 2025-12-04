"""Tests for the LLMling agent."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import ModelResponse, PartDeltaEvent, TextPart, TextPartDelta
from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import Agent, AgentPool, ChatMessage
from llmling_agent.agent.events import StreamCompleteEvent


SIMPLE_PROMPT = "Hello, how are you?"
TEST_RESPONSE = "I am a test response"


async def test_simple_agent_run(test_agent: Agent[None]):
    """Test basic agent text response."""
    result = await test_agent.run(SIMPLE_PROMPT)
    assert isinstance(result.data, str)
    assert result.data == TEST_RESPONSE
    assert result.cost_info is not None


async def test_agent_message_history(test_agent: Agent[None]):
    """Test agent with message history."""
    history = [
        ChatMessage.user_prompt("Previous message"),
        ChatMessage(
            content="Previous response",
            role="assistant",
            messages=[ModelResponse(parts=[TextPart(content="Previous response")])],
        ),
    ]
    test_agent.conversation.set_history(history)
    result = await test_agent.run(SIMPLE_PROMPT)
    assert result.data == TEST_RESPONSE
    assert test_agent.conversation.last_run_messages
    assert len(test_agent.conversation.last_run_messages) == 2  # noqa: PLR2004


async def test_agent_streaming(test_agent: Agent[None]):
    """Test agent streaming response."""
    collected_chunks = []
    final_message = None

    async for event in test_agent.run_stream(SIMPLE_PROMPT):
        match event:
            case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                collected_chunks.append(delta)
            case StreamCompleteEvent(message=message):
                final_message = message

    assert "".join(collected_chunks) == TEST_RESPONSE
    assert final_message is not None
    assert final_message.content == TEST_RESPONSE


async def test_agent_streaming_pydanticai_history(test_agent: Agent[None]):
    """Test streaming pydantic-ai history."""
    history = [
        ChatMessage.user_prompt("Previous message"),
        ChatMessage(
            role="assistant",
            content="Previous response",
            messages=[ModelResponse(parts=[TextPart(content="Previous response")])],
        ),
    ]
    test_agent.conversation.set_history(history)

    collected_chunks = []
    final_message = None

    async for event in test_agent.run_stream(SIMPLE_PROMPT):
        match event:
            case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                collected_chunks.append(delta)
            case StreamCompleteEvent(message=message):
                final_message = message

    result = "".join(collected_chunks)
    assert result == TEST_RESPONSE
    assert final_message is not None
    assert final_message.content == TEST_RESPONSE

    # Check conversation history increased
    messages = test_agent.conversation.get_history()
    assert len(messages) == 4  # Original 2 + new 2  # noqa: PLR2004


async def test_agent_concurrent_runs(test_agent: Agent[None]):
    """Test running multiple prompts concurrently."""
    prompts = ["Hello!", "Hi there!", "Good morning!"]
    tasks = [test_agent.run(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    assert all(r.data == TEST_RESPONSE for r in results)


async def test_agent_model_override():
    """Test overriding model for specific runs."""
    default_response = "default response"
    override_response = "override response"
    model = TestModel(custom_output_text=default_response)
    async with Agent(model=model, name="test-agent") as agent:
        # Run with default model
        result1 = await agent.run(SIMPLE_PROMPT)
        assert result1.data == default_response

        # Run with overridden model
        model2 = TestModel(custom_output_text=override_response)
        result2 = await agent.run(SIMPLE_PROMPT, model=model2)
        assert result2.data == override_response


def test_sync_wrapper(test_agent: Agent[None]):
    """Test synchronous wrapper method."""
    result = test_agent.run.sync(SIMPLE_PROMPT)  # type: ignore[attr-defined]
    assert result.data == TEST_RESPONSE


async def test_agent_forwarding():
    """Test message forwarding between agents."""
    async with AgentPool() as pool:
        model = TestModel(custom_output_text="Main response")
        main_agent = await pool.add_agent("main-agent", model=model)
        model = TestModel(custom_output_text="Helper response")
        helper_agent = await pool.add_agent("helper-agent", model=model)
        main_agent.connect_to(helper_agent)  # Set up forwarding
        messages: list[ChatMessage[Any]] = []  # Track messages from both agents
        main_agent.message_sent.connect(messages.append)
        helper_agent.message_sent.connect(messages.append)
        message = "Hello, agent!"  # Send message and wait for forwarding
        await main_agent.run(message)
        await main_agent.task_manager.complete_tasks()
        await helper_agent.task_manager.complete_tasks()

        # Verify both agents responded
        assert len(messages) == 2  # noqa: PLR2004
        assert any(m.name == "main-agent" for m in messages)
        assert any(m.name == "helper-agent" for m in messages)
        assert any(m.content == "Main response" for m in messages)
        assert any(m.content == "Helper response" for m in messages)
        # Verify metrics are present
        assert all(m.cost_info is not None for m in messages)
        assert all(m.response_time is not None for m in messages)


if __name__ == "__main__":
    pytest.main([__file__])
