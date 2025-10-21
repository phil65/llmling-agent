"""Test contextual progress handler functionality with Agent and TestModel."""

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from llmling_agent import Agent
from llmling_agent_config.mcp_server import StdioMCPServerConfig


# Constants for test expectations
EXPECTED_PROGRESS_EVENTS = 3
PROGRESS_COMPLETION_THRESHOLD = 99
TEST_PROGRESS_VALUE = 50.0


class ProgressCapture:
    """Captures progress callbacks with full context."""

    def __init__(self):
        self.progress_events: list[dict[str, Any]] = []
        self.completed = asyncio.Event()

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        """Capture progress with full context."""
        event = {
            "progress": progress,
            "total": total,
            "message": message,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_input": tool_input,
        }
        self.progress_events.append(event)

        # Signal completion when we reach expected threshold
        if progress >= PROGRESS_COMPLETION_THRESHOLD:
            self.completed.set()


@pytest.mark.asyncio
async def test_contextual_progress_handler_with_agent():
    """Test that progress handlers receive tool context information via Agent."""
    # Setup progress capture
    progress_capture = ProgressCapture()

    # Get server path
    server_path = Path(__file__).parent / "server.py"

    # Create MCP server config
    mcp_server = StdioMCPServerConfig(
        name="progress_test_server",
        command="uv",
        args=["run", str(server_path)],
    )

    # Create agent with TestModel configured to call only test_progress tool
    from pydantic_ai.models.test import TestModel

    test_model = TestModel(call_tools=["test_progress"])

    agent = Agent(
        name="progress_test_agent",
        model=test_model,
        system_prompt="You are a test assistant that calls tools.",
        mcp_servers=[mcp_server],
    )

    async with agent:
        # Wait for MCP servers to initialize
        await asyncio.sleep(0.5)

        # Verify MCP tools are available
        tools = await agent.tools.get_tools()
        tool_names = [tool.name for tool in tools]

        assert "test_progress" in tool_names, (
            f"test_progress tool not found in {tool_names}"
        )

        # Get the MCP manager and patch the progress handler
        mcp_manager = agent.mcp
        assert mcp_manager is not None, "MCP manager should be available"

        # Find the client and set our progress handler
        client = None
        for client_instance in mcp_manager.clients.values():
            client = client_instance
            break

        assert client is not None, "Should have found MCP client"

        # Set our contextual progress handler
        client._contextual_progress_handler = progress_capture

        # Run the agent with a request - TestModel will call the test_progress tool
        # since we configured it with call_tools=["test_progress"]
        await agent.run("Please help me test progress tracking.")

        # Wait for progress events to complete
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(progress_capture.completed.wait(), timeout=15.0)

        # Verify results

        # Verify we captured progress events
        assert len(progress_capture.progress_events) >= EXPECTED_PROGRESS_EVENTS, (
            f"Should have captured at least {EXPECTED_PROGRESS_EVENTS} progress events, "
            f"got {len(progress_capture.progress_events)}"
        )

        # Check that all events have contextual information
        for event in progress_capture.progress_events:
            # Basic FastMCP fields should be present
            assert event["progress"] is not None, "Progress should be set"
            assert event["message"] is not None, "Message should be set"

            # Our contextual fields should be present
            assert event["tool_name"] == "test_progress", (
                f"Tool name should be 'test_progress', got {event['tool_name']}"
            )
            assert event["tool_call_id"] is not None, "Tool call ID should be set"
            assert event["tool_input"] is not None, "Tool input should be set"

            # Tool input should contain message parameter
            tool_input = event["tool_input"]
            assert isinstance(tool_input, dict), "Tool input should be a dict"
            assert "message" in tool_input, "Tool input should have message parameter"

        # Verify progress sequence
        progress_values = [
            event["progress"] for event in progress_capture.progress_events
        ]
        assert progress_values == sorted(progress_values), (
            f"Progress values should be increasing, got {progress_values}"
        )

        # Check specific progress messages from server.py
        messages = [event["message"] for event in progress_capture.progress_events]
        expected_messages = ["first step", "second step", "third step"]
        for expected_msg in expected_messages:
            assert any(expected_msg in str(msg) for msg in messages if msg), (
                f"Should contain '{expected_msg}' in messages: {messages}"
            )


@pytest.mark.asyncio
async def test_progress_handler_without_context():
    """Test that progress handlers work even when context is not available."""
    events = []

    async def simple_progress_handler(
        progress: float,
        total: float | None,
        message: str | None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        """Simple handler that works with or without context."""
        events.append({"progress": progress, "has_context": tool_name is not None})

    # This test verifies the handler can work with just the FastMCP parameters
    await simple_progress_handler(TEST_PROGRESS_VALUE, 100.0, "test", None, None, None)

    assert len(events) == 1
    assert events[0]["progress"] == TEST_PROGRESS_VALUE
    assert events[0]["has_context"] is False


@pytest.mark.asyncio
async def test_direct_mcp_client_progress():
    """Test contextual progress handler with direct MCP client call (no RunContext)."""
    from llmling_agent.mcp_server.client import MCPClient

    # Setup progress capture
    progress_capture = ProgressCapture()

    # Get server path
    server_path = Path(__file__).parent / "server.py"

    # Create MCP server config
    mcp_server = StdioMCPServerConfig(
        name="progress_test_server",
        command="uv",
        args=["run", str(server_path)],
    )

    # Create MCP client directly with our progress handler
    client = MCPClient(progress_handler=progress_capture)

    async with client:
        # Connect to the test server
        await client.connect(mcp_server)

        # Wait a bit for server to be ready
        await asyncio.sleep(0.5)

        # Get available tools to verify connection
        tools = client.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "test_progress" in tool_names, (
            f"test_progress tool not found in {tool_names}"
        )

        # Call the progress test tool directly (fallback path without RunContext)
        test_message = "Testing contextual progress directly"
        await client.call_tool(
            name="test_progress",
            arguments={"message": test_message},
            tool_call_id="test-call-123",
        )

        # Wait for progress to complete
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(progress_capture.completed.wait(), timeout=5.0)

        # Verify we captured progress events (should work with fallback path)
        assert len(progress_capture.progress_events) >= 1, (
            f"Should have captured at least 1 progress event, "
            f"got {len(progress_capture.progress_events)}"
        )

        # Check that events have contextual information from fallback path
        for event in progress_capture.progress_events:
            assert event["progress"] is not None, "Progress should be set"
            assert event["tool_name"] == "test_progress", (
                "Tool name should be test_progress"
            )
            assert event["tool_call_id"] == "test-call-123", "Tool call ID should match"
            assert event["tool_input"] is not None, "Tool input should be set"
            assert event["tool_input"]["message"] == test_message, (
                "Should contain our test message"
            )


if __name__ == "__main__":
    asyncio.run(test_contextual_progress_handler_with_agent())
