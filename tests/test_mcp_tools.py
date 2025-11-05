"""Tests for MCP tools functionality."""

from __future__ import annotations

from pydantic_ai import FunctionToolCallEvent
import pytest

from llmling_agent import Agent


@pytest.mark.flaky(reruns=3)
async def test_mcp_tool_call(default_model: str):
    """Test basic MCP tool functionality with context7 server."""
    sys_prompt = "Look up pydantic docs"
    model = default_model

    # Track tool usage
    tool_calls = []

    def track_tool_usage(event: FunctionToolCallEvent, **kwargs):
        tool_calls.append((event, kwargs))

    async with Agent(model=model, mcp_servers=["npx -y @upstash/context7-mcp"]) as agent:
        agent.run_stream[FunctionToolCallEvent].connect(track_tool_usage)
        events = [i async for i in agent.run_stream(sys_prompt)]
        assert events
        assert len(tool_calls) > 0

        # Verify tools were called (MCP server should be available)
        # Note: This test might be flaky if the MCP server is unavailable
        # In a real test suite, you'd want to mock the MCP server


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
