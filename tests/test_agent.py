"""Tests for the LLMling agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
import pytest
import yamling

from llmling_agent.agent import Agent


if TYPE_CHECKING:
    from pathlib import Path

    from llmling import RuntimeConfig


SIMPLE_PROMPT = "Hello, how are you?"
TEST_RESPONSE = "I am a test response"


@pytest.mark.asyncio
async def test_simple_agent_run(test_agent: Agent[Any, str]):
    """Test basic agent text response."""
    result = await test_agent.run(SIMPLE_PROMPT)
    assert isinstance(result.data, str)
    assert result.data == TEST_RESPONSE
    assert result.cost_info is not None


@pytest.mark.asyncio
async def test_agent_message_history(test_agent: Agent[Any, str]):
    """Test agent with message history."""
    history = [
        ModelRequest(parts=[UserPromptPart(content="Previous message")]),
        ModelResponse(parts=[TextPart(content="Previous response")]),
    ]
    result = await test_agent.run(SIMPLE_PROMPT, message_history=history)
    assert result.data == TEST_RESPONSE
    assert test_agent.conversation.last_run_messages
    assert len(test_agent.conversation.last_run_messages) == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_agent_streaming(test_agent: Agent[Any, str]):
    """Test agent streaming response."""
    stream_ctx = test_agent.run_stream(SIMPLE_PROMPT)
    async with stream_ctx as stream:
        collected = [str(message) async for message in stream.stream()]
        assert "".join(collected) == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_streaming_with_history(test_agent: Agent[Any, str]):
    """Test streaming with message history."""
    history = [
        ModelRequest(parts=[UserPromptPart(content="Previous message")]),
        ModelResponse(parts=[TextPart(content="Previous response")]),
    ]

    stream_ctx = test_agent.run_stream(SIMPLE_PROMPT, message_history=history)
    async with stream_ctx as stream:
        collected = [str(msg) async for msg in stream.stream()]
        result = "".join(collected)
        assert result == TEST_RESPONSE

        # Verify we get the current exchange messages
        new_messages = stream.new_messages()
        assert len(new_messages) == 2  # Current prompt + response  # noqa: PLR2004

        # Check prompt message
        assert isinstance(new_messages[0], ModelRequest)
        assert isinstance(new_messages[0].parts[0], UserPromptPart)
        assert new_messages[0].parts[0].content == SIMPLE_PROMPT

        # Check response message
        assert isinstance(new_messages[1], ModelResponse)
        assert isinstance(new_messages[1].parts[0], TextPart)
        assert new_messages[1].parts[0].content == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_concurrent_runs(test_agent: Agent[Any, str]):
    """Test running multiple prompts concurrently."""
    prompts = ["Hello!", "Hi there!", "Good morning!"]
    tasks = [test_agent.run(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    assert all(r.data == TEST_RESPONSE for r in results)


@pytest.mark.asyncio
async def test_agent_model_override(no_tool_runtime: RuntimeConfig):
    """Test overriding model for specific runs."""
    default_response = "default response"
    override_response = "override response"
    model = TestModel(custom_result_text=default_response)
    agent = Agent[Any, str](runtime=no_tool_runtime, name="test-agent", model=model)

    # Run with default model
    result1 = await agent.run(SIMPLE_PROMPT)
    assert result1.data == default_response

    # Run with overridden model
    model2 = TestModel(custom_result_text=override_response)
    result2 = await agent.run(SIMPLE_PROMPT, model=model2)
    assert result2.data == override_response


def test_sync_wrapper(test_agent: Agent[Any, str]):
    """Test synchronous wrapper method."""
    result = test_agent.run_sync(SIMPLE_PROMPT)
    assert result.data == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_context_manager(tmp_path: Path):
    """Test using agent as async context manager."""
    # Create a minimal config file
    caps = {"load_resource": False, "get_resources": False}
    config = {"global_settings": {"llm_capabilities": caps}}

    # Write config to temporary file
    config_path = tmp_path / "test_config.yml"
    config_path.write_text(yamling.dump_yaml(config))
    model = TestModel(custom_result_text=TEST_RESPONSE)

    async with Agent[Any, str].open(config_path, name="test-agent", model=model) as agent:
        result = await agent.run(SIMPLE_PROMPT)
        assert result.data == TEST_RESPONSE

        # Verify we get expected message sequence
        messages = agent.conversation
        # user prompt -> model response
        assert len(messages) == 2  # noqa: PLR2004

        # Check prompt message
        assert messages[0].content == SIMPLE_PROMPT
        assert messages[1].content == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_logging(no_tool_runtime: RuntimeConfig):
    """Test agent logging functionality."""
    # Test with logging enabled
    agent1 = Agent[Any, str](
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(custom_result_text=TEST_RESPONSE),
        enable_logging=True,
    )
    result1 = await agent1.run(SIMPLE_PROMPT)
    assert result1.data == TEST_RESPONSE

    # Test with logging disabled
    agent2 = Agent[Any, str](
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(custom_result_text=TEST_RESPONSE),
        enable_logging=False,
    )
    result2 = await agent2.run(SIMPLE_PROMPT)
    assert result2.data == TEST_RESPONSE
