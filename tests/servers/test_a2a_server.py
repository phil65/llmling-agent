"""Unit tests for A2AServer."""

from __future__ import annotations

import pytest

from llmling_agent import Agent, AgentPool
from llmling_agent_server.a2a_server import A2AServer


# Test constants
TEST_PORT_BASE = 8001
AGENT_COUNT = 2
EXPECTED_ROUTES = 7  # 2 agents * 3 routes each + 1 root endpoint


@pytest.fixture
def simple_agent_pool():
    """Create a simple agent pool for testing."""

    def callback1(message: str) -> str:
        return f"Agent1: {message}"

    def callback2(message: str) -> str:
        return f"Agent2: {message}"

    agent1 = Agent.from_callback(name="agent1", callback=callback1)
    agent2 = Agent.from_callback(name="agent2", callback=callback2)

    pool = AgentPool()
    pool.register("agent1", agent1)
    pool.register("agent2", agent2)
    return pool


async def test_a2a_server_creation(simple_agent_pool: AgentPool):
    """Test A2AServer can be created with agent pool."""
    server = A2AServer(simple_agent_pool, host="localhost", port=TEST_PORT_BASE)
    assert server.pool is simple_agent_pool
    assert server.host == "localhost"
    assert server.port == TEST_PORT_BASE
    assert len(server.pool.agents) == AGENT_COUNT


async def test_a2a_server_initialization(simple_agent_pool: AgentPool):
    """Test A2AServer initialization with pool context."""
    server = A2AServer(simple_agent_pool, host="localhost", port=TEST_PORT_BASE + 1)

    async with server:
        # Pool should be initialized
        assert server.pool is not None
        assert len(server.pool.agents) == AGENT_COUNT


async def test_a2a_server_base_url(simple_agent_pool: AgentPool):
    """Test A2AServer URL generation."""
    port = TEST_PORT_BASE + 2
    server = A2AServer(simple_agent_pool, host="localhost", port=port)
    assert server.base_url == f"http://localhost:{port}"


async def test_a2a_server_agent_url(simple_agent_pool: AgentPool):
    """Test A2AServer agent URL generation."""
    port = TEST_PORT_BASE + 3
    server = A2AServer(simple_agent_pool, host="localhost", port=port)
    url = server.get_agent_url("agent1")
    assert url == f"http://localhost:{port}/agent1"


async def test_a2a_server_agent_card_url(simple_agent_pool: AgentPool):
    """Test A2AServer agent card URL generation."""
    port = TEST_PORT_BASE + 4
    server = A2AServer(simple_agent_pool, host="localhost", port=port)
    url = server.get_agent_card_url("agent1")
    assert url == f"http://localhost:{port}/agent1/.well-known/agent-card.json"


async def test_a2a_server_list_routes(simple_agent_pool: AgentPool):
    """Test A2AServer route listing."""
    port = TEST_PORT_BASE + 5
    server = A2AServer(simple_agent_pool, host="localhost", port=port)
    routes = server.list_agent_routes()

    assert isinstance(routes, dict)
    assert len(routes) == AGENT_COUNT
    assert "agent1" in routes
    assert "agent2" in routes
    assert routes["agent1"]["endpoint"] == f"http://localhost:{port}/agent1"
    assert (
        routes["agent1"]["agent_card"]
        == f"http://localhost:{port}/agent1/.well-known/agent-card.json"
    )
    assert routes["agent1"]["docs"] == f"http://localhost:{port}/agent1/docs"


async def test_a2a_server_app_creation(simple_agent_pool: AgentPool):
    """Test A2AServer Starlette app creation."""
    pytest.importorskip("starlette")

    server = A2AServer(simple_agent_pool, host="localhost", port=TEST_PORT_BASE + 6)

    async with server:
        routes = await server.get_routes()
        assert routes is not None
        # Should have routes for each agent (3 per agent: endpoint, card, docs) plus root
        assert len(routes) == EXPECTED_ROUTES


@pytest.mark.skip(reason="Requires specific YAML schema - integration test")
async def test_a2a_server_from_config(tmp_path):
    """Test A2AServer creation from config file."""
    pytest.importorskip("starlette")

    # Create a minimal config file
    config_content = """
    environments:
      test:
        type: inline
        module_path: llmling_agent
        class_name: Agent

    agents:
      test_agent:
        name: test_agent
        environment: test
        model: test
        system_prompts:
          - You are a test agent
    """

    config_path = tmp_path / "test_config.yml"
    config_path.write_text(config_content)

    port = TEST_PORT_BASE + 7
    server = A2AServer.from_config(
        config_path,
        host="localhost",
        port=port,
    )

    assert server.host == "localhost"
    assert server.port == port
    assert len(server.pool.agents) >= 1


async def test_a2a_server_name_generation(simple_agent_pool: AgentPool):
    """Test A2AServer auto-generates name when not provided."""
    server = A2AServer(simple_agent_pool)
    assert server.name is not None
    assert "A2AServer" in server.name


async def test_a2a_server_custom_name(simple_agent_pool: AgentPool):
    """Test A2AServer uses custom name when provided."""
    server = A2AServer(simple_agent_pool, name="custom-a2a-server")
    assert server.name == "custom-a2a-server"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
