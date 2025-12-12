"""Unit tests for AggregatingServer."""

from __future__ import annotations

import pytest

from llmling_agent import Agent, AgentPool
from llmling_agent_server import A2AServer, AggregatingServer, AGUIServer
from llmling_agent_server.http_server import HTTPServer


# Test constants
TEST_PORT_BASE = 9000
AGENT_COUNT = 2
SERVER_COUNT = 2
INITIALIZED_COUNT = 2


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


@pytest.fixture
def agui_server(simple_agent_pool: AgentPool):
    """Create AGUIServer for testing."""
    return AGUIServer(simple_agent_pool, host="localhost", port=TEST_PORT_BASE)


@pytest.fixture
def a2a_server(simple_agent_pool: AgentPool):
    """Create A2AServer for testing."""
    return A2AServer(simple_agent_pool, host="localhost", port=TEST_PORT_BASE + 1)


async def test_aggregating_server_creation(simple_agent_pool: AgentPool):
    """Test AggregatingServer can be created with multiple servers."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 10)
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 11)

    server = AggregatingServer(simple_agent_pool, servers=[agui, a2a])

    assert server.pool is simple_agent_pool
    assert len(server.servers) == SERVER_COUNT


async def test_aggregating_server_requires_servers(simple_agent_pool: AgentPool):
    """Test AggregatingServer requires at least one server."""
    with pytest.raises(ValueError, match="At least one server must be provided"):
        AggregatingServer(simple_agent_pool, servers=[])


async def test_aggregating_server_separate_mode(simple_agent_pool: AgentPool):
    """Test AggregatingServer in separate mode (default)."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 20)
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 21)

    server = AggregatingServer(simple_agent_pool, servers=[agui, a2a])

    assert server.unified_http is False
    assert "separate" in repr(server)


async def test_aggregating_server_unified_mode_creation(simple_agent_pool: AgentPool):
    """Test AggregatingServer in unified HTTP mode."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 30)
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 31)

    server = AggregatingServer(
        simple_agent_pool,
        servers=[agui, a2a],
        unified_http=True,
        unified_host="localhost",
        unified_port=TEST_PORT_BASE + 32,
    )

    assert server.unified_http is True
    assert server.unified_host == "localhost"
    assert server.unified_port == TEST_PORT_BASE + 32
    assert "unified" in repr(server)


async def test_aggregating_server_unified_base_url(simple_agent_pool: AgentPool):
    """Test AggregatingServer unified base URL."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 40)

    server = AggregatingServer(
        simple_agent_pool,
        servers=[agui],
        unified_http=True,
        unified_host="localhost",
        unified_port=TEST_PORT_BASE + 41,
    )

    expected_url = f"http://localhost:{TEST_PORT_BASE + 41}"
    assert server.unified_base_url == expected_url


async def test_aggregating_server_initialization(simple_agent_pool: AgentPool):
    """Test AggregatingServer initialization with pool context."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 50)
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 51)

    server = AggregatingServer(simple_agent_pool, servers=[agui, a2a])

    async with server:
        assert server.initialized_server_count == SERVER_COUNT
        assert len(server._initialized_servers) == SERVER_COUNT


async def test_aggregating_server_list_servers(simple_agent_pool: AgentPool):
    """Test AggregatingServer server listing."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 60, name="agui-test")
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 61, name="a2a-test")

    server = AggregatingServer(simple_agent_pool, servers=[agui, a2a])

    servers_info = server.list_servers()
    assert len(servers_info) == SERVER_COUNT

    names = [s.name for s in servers_info]
    assert "agui-test" in names
    assert "a2a-test" in names

    # Check is_http flag
    for info in servers_info:
        assert info.is_http is True  # Both are HTTPServer subclasses


async def test_aggregating_server_get_server(simple_agent_pool: AgentPool):
    """Test AggregatingServer get_server method."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 70, name="agui-find")
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 71, name="a2a-find")

    server = AggregatingServer(simple_agent_pool, servers=[agui, a2a])

    found = server.get_server("agui-find")
    assert found is not None
    assert found.name == "agui-find"

    not_found = server.get_server("nonexistent")
    assert not_found is None


async def test_aggregating_server_add_remove_server(simple_agent_pool: AgentPool):
    """Test adding and removing servers from AggregatingServer."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 80, name="agui-add")

    server = AggregatingServer(simple_agent_pool, servers=[agui])
    initial_count = 1
    assert len(server.servers) == initial_count

    # Add server
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 81, name="a2a-add")
    server.add_server(a2a)
    assert len(server.servers) == initial_count + 1

    # Remove server
    server.remove_server(a2a)
    assert len(server.servers) == initial_count


async def test_aggregating_server_is_http_server_detection(simple_agent_pool: AgentPool):
    """Test that AggregatingServer correctly detects HTTP servers."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 90)

    server = AggregatingServer(simple_agent_pool, servers=[agui])

    async with server:
        http_servers = server._get_http_servers()
        expected_http_count = 1
        assert len(http_servers) == expected_http_count
        assert isinstance(http_servers[0], HTTPServer)


async def test_aggregating_server_collect_routes_unified(simple_agent_pool: AgentPool):
    """Test route collection in unified mode."""
    pytest.importorskip("starlette")

    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 100)
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 101)

    server = AggregatingServer(
        simple_agent_pool,
        servers=[agui, a2a],
        unified_http=True,
        unified_port=TEST_PORT_BASE + 102,
    )

    async with server:
        routes = await server._collect_all_routes()

        # Should have routes from both servers with prefixes
        route_paths = [r.path for r in routes]

        # Check AGUI routes have /agui prefix
        agui_routes = [p for p in route_paths if p.startswith("/agui")]
        assert agui_routes  # At least one route

        # Check A2A routes have /a2a prefix
        a2a_routes = [p for p in route_paths if p.startswith("/a2a")]
        assert a2a_routes  # At least one route


async def test_aggregating_server_create_unified_app(simple_agent_pool: AgentPool):
    """Test unified app creation."""
    pytest.importorskip("starlette")

    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 110)
    a2a = A2AServer(simple_agent_pool, port=TEST_PORT_BASE + 111)

    server = AggregatingServer(
        simple_agent_pool,
        servers=[agui, a2a],
        unified_http=True,
        unified_port=TEST_PORT_BASE + 112,
    )

    async with server:
        app = await server._create_unified_app()
        assert app is not None

        # App should have routes from both servers plus root
        assert app.routes  # At least one route


async def test_aggregating_server_status(simple_agent_pool: AgentPool):
    """Test server status tracking."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 120, name="status-test")

    server = AggregatingServer(simple_agent_pool, servers=[agui])

    # Before initialization
    status = server.get_server_status()
    assert status["status-test"] == "not_initialized"

    async with server:
        # After initialization
        status = server.get_server_status()
        assert status["status-test"] == "initialized"


async def test_aggregating_server_repr(simple_agent_pool: AgentPool):
    """Test AggregatingServer string representation."""
    agui = AGUIServer(simple_agent_pool, port=TEST_PORT_BASE + 130)

    server = AggregatingServer(
        simple_agent_pool,
        servers=[agui],
        name="test-aggregator",
    )

    repr_str = repr(server)
    assert "AggregatingServer" in repr_str
    assert "test-aggregator" in repr_str
    assert "servers=1" in repr_str


if __name__ == "__main__":
    pytest.main(["-v", __file__])
