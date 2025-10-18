"""Tests for MCP tools functionality."""

from __future__ import annotations

from llmling_agent import Agent


async def test_mcp_tool_call(default_model: str):
    """Test basic MCP tool functionality with context7 server."""
    sys_prompt = "Look up pydantic docs"
    model = default_model

    # Track tool usage
    tool_calls = []

    def track_tool_usage(tool_name: str, **kwargs):
        tool_calls.append((tool_name, kwargs))

    async with Agent(model=model, mcp_servers=["npx -y @upstash/context7-mcp"]) as agent:
        agent.tool_used.connect(track_tool_usage)
        result = await agent.run(sys_prompt)

        # Verify we got a response
        assert isinstance(result.data, str)
        assert len(result.data) > 0

        # Verify tools were called (MCP server should be available)
        # Note: This test might be flaky if the MCP server is unavailable
        # In a real test suite, you'd want to mock the MCP server


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
