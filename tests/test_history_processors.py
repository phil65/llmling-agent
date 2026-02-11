from __future__ import annotations

import inspect
from unittest.mock import patch

from pydantic_ai.models.test import TestModel
import pytest

from agentpool import Agent
from agentpool_config.session import MemoryConfig


@pytest.fixture
def mock_model():
    return TestModel(custom_output_text="Response")


def test_config_validation_empty():
    """Test MemoryConfig with empty processors list."""
    config = MemoryConfig(history_processors=[])
    assert config.history_processors == []


def test_config_validation_none():
    """Test MemoryConfig with None processors."""
    config = MemoryConfig(history_processors=None)
    assert config.history_processors is None


async def test_processor_resolution_invalid_path(mock_model):
    """Test resolution with invalid import path."""
    memory_cfg = MemoryConfig(history_processors=["invalid.module:func"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        with pytest.raises(ValueError, match="Failed to resolve history processor"):
            await agent.get_agentlet(None, str)


async def test_processor_resolution_not_callable(mock_model):
    """Test resolution with non-callable import."""
    memory_cfg = MemoryConfig(history_processors=["tests.test_processors:__doc__"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        with pytest.raises(ValueError, match=r"isn't callable|is not callable"):
            await agent.get_agentlet(None, str)


async def test_processor_resolution_invalid_signature_too_many(mock_model):
    """Test resolution with invalid signature (too many args)."""
    memory_cfg = MemoryConfig(
        history_processors=["tests.test_processors:invalid_processor_too_many"]
    )
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        with pytest.raises(ValueError, match="must take 1 or 2 arguments"):
            await agent.get_agentlet(None, str)


async def test_processor_resolution_invalid_signature_wrong_name(mock_model):
    """Test resolution with invalid signature (wrong name)."""
    memory_cfg = MemoryConfig(
        history_processors=["tests.test_processors:invalid_processor_wrong_name"]
    )
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        with pytest.raises(ValueError, match="must be messages/msgs/history"):
            await agent.get_agentlet(None, str)


async def test_processor_resolution_sync_no_ctx(mock_model):
    """Test resolution of sync processor without context."""
    memory_cfg = MemoryConfig(history_processors=["tests.test_processors:keep_recent"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        processors = await agent.get_agentlet(None, str)
        assert len(processors.history_processors) == 1
        assert processors.history_processors[0].__name__ == "keep_recent"


async def test_processor_resolution_async_no_ctx(mock_model):
    """Test resolution of async processor without context."""
    memory_cfg = MemoryConfig(history_processors=["tests.test_processors:filter_thinking_async"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        processors = await agent.get_agentlet(None, str)
        assert len(processors.history_processors) == 1
        assert processors.history_processors[0].__name__ == "filter_thinking_async"
        assert inspect.iscoroutinefunction(processors.history_processors[0])


async def test_processor_resolution_sync_ctx(mock_model):
    """Test resolution of sync processor with context."""
    memory_cfg = MemoryConfig(history_processors=["tests.test_processors:context_aware_sync"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        processors = await agent.get_agentlet(None, str)
        assert len(processors.history_processors) == 1
        assert processors.history_processors[0].__name__ == "context_aware_sync"


async def test_processor_resolution_async_ctx(mock_model):
    """Test resolution of async processor with context."""
    memory_cfg = MemoryConfig(history_processors=["tests.test_processors:context_aware_async"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        processors = await agent.get_agentlet(None, str)
        assert len(processors.history_processors) == 1
        assert processors.history_processors[0].__name__ == "context_aware_async"


async def test_processor_caching(mock_model):
    """Test that processors are only resolved and imported once."""
    memory_cfg = MemoryConfig(history_processors=["tests.test_processors:keep_recent"])
    async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
        with patch(
            "agentpool.utils.importing.import_callable", side_effect=lambda x: lambda y: y
        ) as mock_import:
            # First call
            await agent.get_agentlet(None, str)
            assert mock_import.call_count == 1

            # Second call - should use cache
            await agent.get_agentlet(None, str)
            assert mock_import.call_count == 1


async def test_integration_processors_called(mock_model):
    """Integration test: Verify processor is actually called during run."""
    called = False

    def my_processor(messages):
        nonlocal called
        called = True
        return messages

    with patch("agentpool.utils.importing.import_callable", return_value=my_processor):
        memory_cfg = MemoryConfig(history_processors=["mock:processor"])
        async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
            await agent.run("Hello")
            assert called is True


async def test_multiple_processors_sequential(mock_model):
    """Verify multiple processors run in sequence."""
    order = []

    def proc1(messages):
        order.append(1)
        return messages

    def proc2(messages):
        order.append(2)
        return messages

    with patch("agentpool.utils.importing.import_callable", side_effect=[proc1, proc2]):
        memory_cfg = MemoryConfig(history_processors=["p1", "p2"])
        async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
            await agent.run("Hello")
            assert order == [1, 2]


async def test_compatibility_no_processors(mock_model):
    """Verify agent works fine without processors."""
    async with Agent(name="test", model=mock_model) as agent:
        agentlet = await agent.get_agentlet(None, str)
        assert agentlet.history_processors == []
        result = await agent.run("Hello")
        assert result.data == "Response"


async def test_history_processor_with_existing_history(mock_model):
    """Test that history processors receive all messages including existing history."""
    from pydantic_ai import ModelResponse, TextPart

    from agentpool.messaging import ChatMessage

    # Setup history with 4 messages
    history = [
        ChatMessage.user_prompt("M1"),
        ChatMessage(
            role="assistant", content="R1", messages=[ModelResponse(parts=[TextPart(content="R1")])]
        ),
        ChatMessage.user_prompt("M2"),
        ChatMessage(
            role="assistant", content="R2", messages=[ModelResponse(parts=[TextPart(content="R2")])]
        ),
    ]

    seen_messages = []

    def my_processor(messages):
        nonlocal seen_messages
        seen_messages = messages
        return messages

    # Configure processor to record what it sees
    memory_cfg = MemoryConfig(history_processors=["mock:processor"])

    with patch("agentpool.utils.importing.import_callable", return_value=my_processor):
        async with Agent(name="test", model=mock_model, session=memory_cfg) as agent:
            agent.conversation.set_history(history)
            await agent.run("Hello")

            # Processor should see all 4 history messages + 1 new user message = 5 total
            assert len(seen_messages) == 5
            # Check that all messages are present
            assert "M1" in str(seen_messages[0])
            assert "R1" in str(seen_messages[1])
            assert "M2" in str(seen_messages[2])
            assert "R2" in str(seen_messages[3])
            assert "Hello" in str(seen_messages[4])
