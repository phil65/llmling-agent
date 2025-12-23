"""Test fixtures for ACP tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from acp import ClientCapabilities, DefaultACPClient, FileSystemCapability
from acp.agent.implementations import TestAgent
from agentpool_server.acp_server.acp_agent import AgentPoolACPAgent


@pytest.fixture
def test_client() -> DefaultACPClient:
    """Create a fresh test client for each test."""
    return DefaultACPClient(allow_file_operations=True, use_real_files=False)


@pytest.fixture
def test_agent() -> TestAgent:
    """Create a fresh test agent for each test."""
    return TestAgent()


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return Mock()


@pytest.fixture
def mock_agent_pool():
    """Create a mock agent pool with a test agent."""
    from agentpool import Agent
    from agentpool.delegation import AgentPool

    # Create a simple test agent
    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    agent = Agent.from_callback(name="test_agent", callback=simple_callback)
    pool = AgentPool()
    pool.register("test_agent", agent)
    return pool


@pytest.fixture
def client_capabilities():
    """Create client capabilities for testing."""
    fs_caps = FileSystemCapability(read_text_file=True, write_text_file=True)
    return ClientCapabilities(fs=fs_caps, terminal=True)


@pytest.fixture
def mock_acp_agent(mock_connection, mock_agent_pool, client_capabilities) -> AgentPoolACPAgent:
    """Create a mock ACP agent for testing."""
    return AgentPoolACPAgent(connection=mock_connection, agent_pool=mock_agent_pool)
