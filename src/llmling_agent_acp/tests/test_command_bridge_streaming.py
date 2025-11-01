"""Tests for streaming command bridge functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from slashed import CommandStore

from llmling_agent import Agent, AgentPool
from llmling_agent_acp.command_bridge import ACPCommandBridge, ACPOutputWriter
from llmling_agent_acp.session import ACPSession


@pytest.mark.asyncio
async def test_acp_output_writer():
    """Test that ACPOutputWriter sends updates immediately to session."""
    # Mock session with notifications
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.notifications = AsyncMock()

    received_messages = []

    async def capture_message(message):
        received_messages.append(message)

    mock_session.notifications.send_agent_text.side_effect = capture_message

    writer = ACPOutputWriter(mock_session)

    # Test sending multiple messages
    await writer.print("First message")
    await writer.print("Second message")

    # Verify messages were sent immediately
    expected_message_count = 2
    assert len(received_messages) == expected_message_count

    # Verify content was sent correctly
    assert received_messages[0] == "First message"
    assert received_messages[1] == "Second message"


@pytest.mark.asyncio
async def test_command_bridge_immediate_execution():
    """Test that command execution sends updates immediately."""

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    # Set up agent and session
    agent = Agent(name="test_agent", provider=simple_callback)
    agent_pool = AgentPool[None]()
    agent_pool.register("test_agent", agent)

    # Create command store and bridge
    command_store = CommandStore()
    command_bridge = ACPCommandBridge(command_store)

    # Mock session with notifications
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.agent = AsyncMock()
    mock_session.agent.context = None
    mock_session.notifications = AsyncMock()

    # Capture messages sent via notifications
    sent_messages = []

    async def capture_message(message):
        sent_messages.append(message)

    mock_session.notifications.send_agent_text.side_effect = capture_message

    # Test command execution
    await command_bridge.execute_slash_command("/help", mock_session)

    # Verify messages were sent immediately via notifications
    assert len(sent_messages) > 0


@pytest.mark.asyncio
async def test_immediate_send_with_slow_command():
    """Test immediate sending works with commands that produce output over time."""

    # Create a command that outputs multiple lines with delays
    async def slow_command_func(ctx, args, kwargs):
        await ctx.print("Starting task...")
        await asyncio.sleep(0.01)  # Small delay
        await ctx.print("Processing...")
        await asyncio.sleep(0.01)  # Small delay
        await ctx.print("Completed!")

    # Set up command store
    command_store = CommandStore()
    command_store.add_command(
        name="slow", fn=slow_command_func, description="A slow command for testing"
    )

    command_bridge = ACPCommandBridge(command_store)

    # Mock session with notifications
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.agent = AsyncMock()
    mock_session.agent.context = None
    mock_session.notifications = AsyncMock()

    # Collect messages with timestamps to verify immediate sending
    messages_with_time = []
    start_time = asyncio.get_event_loop().time()

    async def capture_with_time(message):
        current_time = asyncio.get_event_loop().time()
        messages_with_time.append((message, current_time - start_time))

    mock_session.notifications.send_agent_text.side_effect = capture_with_time

    # Execute command
    await command_bridge.execute_slash_command("/slow", mock_session)

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


@pytest.mark.asyncio
async def test_immediate_send_error_handling():
    """Test that errors in commands are properly sent immediately."""

    async def failing_command(ctx, args, kwargs):
        await ctx.print("Starting...")
        msg = "Command failed!"
        raise ValueError(msg)

    command_store = CommandStore()
    command_store.add_command(
        name="fail", fn=failing_command, description="A failing command"
    )

    command_bridge = ACPCommandBridge(command_store)

    # Mock session with notifications
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.agent = AsyncMock()
    mock_session.agent.context = None
    mock_session.notifications = AsyncMock()

    # Collect all messages
    sent_messages = []

    async def capture_message(message):
        sent_messages.append(message)

    mock_session.notifications.send_agent_text.side_effect = capture_message

    # Execute failing command
    await command_bridge.execute_slash_command("/fail", mock_session)

    # Should get the initial output plus error message
    min_expected_messages = 2
    assert len(sent_messages) >= min_expected_messages

    # Check that we got both normal output and error
    message_text = " ".join(sent_messages)
    assert "Starting..." in message_text
    assert "Command error:" in message_text
