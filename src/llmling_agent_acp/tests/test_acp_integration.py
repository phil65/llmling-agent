"""Proper integration tests for ACP functionality."""

from __future__ import annotations

import tempfile

import pytest


class TestACPIntegration:
    """Test ACP functionality with real components."""

    @pytest.fixture
    async def agent_pool(self):
        """Create a real agent pool from config."""
        from llmling_agent import Agent
        from llmling_agent.delegation import AgentPool

        # Create a simple test agent
        def simple_callback(message: str) -> str:
            return f"Test response: {message}"

        agent = Agent(name="test_agent", provider=simple_callback)
        pool = AgentPool[None]()
        pool.register("test_agent", agent)
        return pool

    async def test_acp_server_creation(self, agent_pool):
        """Test that ACP server can be created from agent pool."""
        from llmling_agent_acp import ACPServer

        server = ACPServer(agent_pool=agent_pool)
        assert server.agent_pool is agent_pool
        assert len(server.agent_pool.agents) > 0

    async def test_filesystem_provider_tool_creation(self, agent_pool):
        """Test that filesystem provider creates tools correctly."""
        from unittest.mock import Mock

        from acp.schema import ClientCapabilities, FileSystemCapability
        from llmling_agent_acp.acp_tools import ACPFileSystemProvider
        from llmling_agent_acp.session import ACPSession

        # Set up session with file capabilities
        mock_client = Mock()
        mock_acp_agent = Mock()

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
        provider = ACPFileSystemProvider(
            session=session,
            client_capabilities=capabilities,
        )

        # Test tool creation
        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}

        # Verify expected tools are created
        assert "read_text_file" in tool_names
        assert "write_text_file" in tool_names

        # Verify tools have correct session reference
        assert provider.session_id == "file-test"
        assert provider.agent is mock_acp_agent

    async def test_agent_switching_workflow(self, agent_pool):
        """Test the complete agent switching workflow."""
        from unittest.mock import Mock

        from acp.schema import ClientCapabilities

        # Add another agent to the pool for switching
        from llmling_agent import Agent
        from llmling_agent.delegation import AgentPool
        from llmling_agent_acp.session import ACPSession

        def callback1(message: str) -> str:
            return f"Agent1 response: {message}"

        def callback2(message: str) -> str:
            return f"Agent2 response: {message}"

        agent1 = Agent(name="agent1", provider=callback1)
        agent2 = Agent(name="agent2", provider=callback2)

        multi_pool = AgentPool[None]()
        multi_pool.register("agent1", agent1)
        multi_pool.register("agent2", agent2)
        mock_client = Mock()
        mock_acp_agent = Mock()
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
