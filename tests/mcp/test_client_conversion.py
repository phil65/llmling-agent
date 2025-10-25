"""Integration test for MCP client conversion functionality.

Tests that our MCP client properly converts FastMCP server responses
to PydanticAI-compatible return types without mocks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai import BinaryContent
import pytest

from llmling_agent.mcp_server.client import MCPClient
from llmling_agent_config.mcp_server import StdioMCPServerConfig


@pytest.fixture
async def mcp_client():
    """Create MCP client connected to test server."""
    server_path = Path(__file__).parent / "server.py"

    config = StdioMCPServerConfig(
        name="test_server",
        command="uv",
        args=["run", str(server_path)],
    )

    client = MCPClient(config)
    await client.connect(config)
    async with client:
        # Wait for server to be ready
        await asyncio.sleep(0.5)
        yield client


async def test_rich_content_image(mcp_client: MCPClient):
    """Test that FastMCP Image content is converted to PydanticAI types."""
    result = await mcp_client.call_tool("test_rich_content", {"content_type": "image"})
    assert isinstance(result.content[0], BinaryContent)
    result = await mcp_client.call_tool("test_rich_content", {"content_type": "audio"})
    assert isinstance(result.content[0], BinaryContent)
    assert result.content[0].media_type == "audio/wav"
    result = await mcp_client.call_tool("test_rich_content", {"content_type": "file"})
    assert result is not None
    result = await mcp_client.call_tool("test_rich_content", {"content_type": "mixed"})
    assert result is not None


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
