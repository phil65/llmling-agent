"""Tests for streaming command functionality in sessions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from acp import ClientCapabilities
from llmling_agent import Agent, AgentPool
from llmling_agent_server.acp_server.session import ACPSession


async def test_session_command_immediate_execution():
    """Test that command execution sends updates immediately."""

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback)
    agent_pool = AgentPool()
    agent_pool.register("test_agent", agent)

    mock_client = AsyncMock()
    mock_acp_agent = AsyncMock()

    session = ACPSession(
        session_id="test_session",
        agent_pool=agent_pool,
        current_agent_name="test_agent",
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent,
        client_capabilities=ClientCapabilities(fs=None, terminal=False),
    )

    sent_messages = []

    async def capture_message(message):
        sent_messages.append(message)

    session.notifications.send_agent_text = capture_message
    await session.execute_slash_command("/help")
    assert len(sent_messages) > 0


async def test_immediate_send_with_slow_command():
    """Test immediate sending works with commands that produce output over time."""

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback)
    agent_pool = AgentPool()
    agent_pool.register("test_agent", agent)

    mock_client = AsyncMock()
    mock_acp_agent = AsyncMock()

    session = ACPSession(
        session_id="test_session",
        agent_pool=agent_pool,
        current_agent_name="test_agent",
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent,
        client_capabilities=ClientCapabilities(fs=None, terminal=False),
    )

    # Create a command that outputs multiple lines with delays
    async def slow_command_func(ctx, args, kwargs):
        await ctx.print("Starting task...")
        await asyncio.sleep(0.01)  # Small delay
        await ctx.print("Processing...")
        await asyncio.sleep(0.01)  # Small delay
        await ctx.print("Completed!")

    # Add the command to the session's command store
    session.command_store.add_command(
        name="slow", fn=slow_command_func, description="A slow command for testing"
    )

    # Collect messages with timestamps to verify immediate sending
    messages_with_time = []
    start_time = asyncio.get_event_loop().time()

    async def capture_with_time(message):
        current_time = asyncio.get_event_loop().time()
        messages_with_time.append((message, current_time - start_time))

    session.notifications.send_agent_text = capture_with_time
    await session.execute_slash_command("/slow")
    # Verify we got multiple messages
    min_expected_messages = 3
    assert len(messages_with_time) >= min_expected_messages
    # Verify messages came at different times (immediate sending behavior)
    times = [time for _, time in messages_with_time]
    assert times[1] > times[0]  # Second message came after first
    assert times[2] > times[1]  # Third message came after second

    # Verify message content is correct
    expected_messages = ["Starting task...", "Processing...", "Completed!"]
    actual_messages = [message for message, _ in messages_with_time]
    for expected in expected_messages:
        assert expected in actual_messages


async def test_immediate_send_error_handling():
    """Test that errors in commands are properly sent immediately."""

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback)
    agent_pool = AgentPool()
    agent_pool.register("test_agent", agent)
    mock_client = AsyncMock()
    mock_acp_agent = AsyncMock()
    session = ACPSession(
        session_id="test_session",
        agent_pool=agent_pool,
        current_agent_name="test_agent",
        cwd="/tmp",
        client=mock_client,
        acp_agent=mock_acp_agent,
        client_capabilities=ClientCapabilities(fs=None, terminal=False),
    )

    async def failing_command(ctx, args, kwargs):
        await ctx.print("Starting...")
        msg = "Command failed!"
        raise ValueError(msg)

    session.command_store.add_command(
        name="fail", fn=failing_command, description="A failing command"
    )

    # Collect all messages
    sent_messages = []

    async def capture_message(message):
        sent_messages.append(message)

    session.notifications.send_agent_text = capture_message
    # Execute failing command
    await session.execute_slash_command("/fail")

    # Should get the initial output plus error message
    min_expected_messages = 2
    assert len(sent_messages) >= min_expected_messages

    # Check that we got both normal output and error
    message_text = " ".join(sent_messages)
    assert "Starting..." in message_text
    assert "Command error:" in message_text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
