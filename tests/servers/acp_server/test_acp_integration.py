"""Proper integration tests for ACP functionality."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock

import pytest

from acp import ClientCapabilities, FileSystemCapability

# Add another agent to the pool for switching
from llmling_agent import Agent
from llmling_agent.delegation import AgentPool
from llmling_agent_server.acp_server import ACPServer
from llmling_agent_server.acp_server.acp_tools import ACPFileSystemProvider
from llmling_agent_server.acp_server.session import ACPSession


class TestACPIntegration:
    """Test ACP functionality with real components."""

    @pytest.fixture
    async def agent_pool(self):
        """Create a real agent pool from config."""

        # Create a simple test agent
        def simple_callback(message: str) -> str:
            return f"Test response: {message}"

        agent = Agent.from_callback(name="test_agent", callback=simple_callback)
        pool = AgentPool()
        pool.register("test_agent", agent)
        return pool

    async def test_acp_server_creation(self, agent_pool):
        """Test that ACP server can be created from agent pool."""
        server = ACPServer(pool=agent_pool)
        assert server.pool is agent_pool
        assert len(server.pool.agents) > 0

    async def test_filesystem_provider_tool_creation(self, agent_pool, mock_acp_agent):
        """Test that filesystem provider creates tools correctly."""
        # Set up session with file capabilities
        mock_client = AsyncMock()

        fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
        capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

        session = ACPSession(
            session_id="file-test",
            agent_pool=agent_pool,
            current_agent_name="test_agent",
            cwd=tempfile.gettempdir(),
            client=mock_client,
            acp_agent=mock_acp_agent,
            client_capabilities=capabilities,
        )

        # Create filesystem provider
        provider = ACPFileSystemProvider(session=session)

        # Test tool creation
        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}

        # Verify expected tools are created
        assert "read_text_file" in tool_names
        assert "write_text_file" in tool_names

        # Verify tools have correct session reference
        assert provider.session_id == "file-test"
        assert provider.agent is mock_acp_agent

    async def test_agent_switching_workflow(self, agent_pool, mock_acp_agent):
        """Test the complete agent switching workflow."""

        def callback1(message: str) -> str:
            return f"Agent1 response: {message}"

        def callback2(message: str) -> str:
            return f"Agent2 response: {message}"

        agent1 = Agent.from_callback(name="agent1", callback=callback1)
        agent2 = Agent.from_callback(name="agent2", callback=callback2)

        multi_pool = AgentPool()
        multi_pool.register("agent1", agent1)
        multi_pool.register("agent2", agent2)
        mock_client = AsyncMock()
        capabilities = ClientCapabilities(fs=None, terminal=False)

        session = ACPSession(
            session_id="switching-test",
            agent_pool=multi_pool,
            current_agent_name="agent1",
            cwd=tempfile.gettempdir(),
            client=mock_client,
            acp_agent=mock_acp_agent,
            client_capabilities=capabilities,
        )

        # Should start with agent1
        assert session.agent.name == "agent1"
        assert session.current_agent_name == "agent1"

        # Switch to agent2
        await session.switch_active_agent("agent2")
        assert session.agent.name == "agent2"
        assert session.current_agent_name == "agent2"

        # Switching to non-existent agent should fail
        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            await session.switch_active_agent("nonexistent")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
