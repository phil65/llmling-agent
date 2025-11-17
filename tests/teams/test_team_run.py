from __future__ import annotations

import asyncio
import functools
from typing import Any

from llmling_models import function_to_model
import pytest

from llmling_agent import Agent, AgentPool, ChatMessage
from llmling_agent.utils.now import get_now


async def delayed_processor(msg: str, delay: float = 0.1) -> str:
    """Test processor that simulates work with a delay."""
    await asyncio.sleep(delay)
    return f"Processed: {msg}"


async def test_single_execution():
    """Test single background execution."""
    async with AgentPool() as pool:
        # Create agents with delayed processors
        agent1 = await pool.add_agent(
            "agent1",
            model=function_to_model(functools.partial(delayed_processor, delay=0.1)),
        )
        agent2 = await pool.add_agent(
            "agent2",
            model=function_to_model(functools.partial(delayed_processor, delay=0.2)),
        )

        run = agent1 | agent2
        input_text = "test message"

        # Start background execution and get stats - now with await
        stats = await run.run_in_background(input_text)
        assert run.is_busy()

        # Wait for completion and get final message
        result = await run.wait()
        assert not run.is_busy()

        # Verify result
        assert isinstance(result, ChatMessage)
        assert result.content.startswith("Processed:")
        # Should be from the last agent in the chain
        assert result.name == "agent2"

        # Verify stats captured all messages
        messages: list[ChatMessage[Any]] = []
        for talk in stats:
            messages.extend(talk.stats.messages)
        assert len(messages) == 2  # One from each agent  # noqa: PLR2004


# async def test_continuous_execution():
#     """Test continuous background execution."""
#     async with AgentPool() as pool:
#         callback = functools.partial(delayed_processor, delay=0.1)
#         agent1 = Agent.from_callback(callback, name="agent1")
#         pool.register(agent1.name, agent1)
#         _stats = await agent1.run_in_background("test", max_count=3, interval=0.1)
#         # Count executions through stats
#         execution_count = 0
#         while agent1.is_busy():
#             print(agent1._background_task)
#             await asyncio.sleep(0.1)
#             # execution_count = len(stats[0].stats.messages)

#         # Wait should return last message
#         result = await agent1.wait()
#         assert execution_count == 3
#         assert isinstance(result, ChatMessage)
#         assert result.content.startswith("Processed:")
#         assert result.name == "agent1"


async def test_error_handling():
    """Test handling of errors in background execution."""

    async def failing_processor(msg: str) -> str:
        await asyncio.sleep(0.1)
        msg = "Test error"
        raise ValueError(msg)

    async with AgentPool() as pool:
        agent = Agent.from_callback(failing_processor, name="failing_agent")
        pool.register(agent.name, agent)

        run = agent
        _stats = await run.run_in_background("test", max_count=1)
        # await asyncio.sleep(1)
        # Should return None if execution failed
        result = await run.wait()
        assert result is None


async def test_cancellation():
    """Test cancellation of background execution."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(
            "agent",
            model=function_to_model(functools.partial(delayed_processor, delay=0.5)),
        )

        run = agent
        _stats = await run.run_in_background(
            "test",
            max_count=None,  # Run indefinitely
        )

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel execution
        await run.stop()
        assert not run.is_busy()

        # Should not be able to wait() after cancellation
        with pytest.raises(RuntimeError):
            await run.wait()


async def test_timing_accuracy():
    """Test that timing information is accurate."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(
            "agent",
            model=function_to_model(functools.partial(delayed_processor, delay=0.2)),
        )

        run = agent
        start = get_now()
        _stats = await run.run_in_background("test", max_count=1)

        # Wait should return message
        result = await run.wait()
        assert isinstance(result, ChatMessage)
        # Message should have timestamp
        assert result.timestamp >= start
        assert result.timestamp < get_now()


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
