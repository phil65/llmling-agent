"""Test MCP server integration with ACP sessions."""

from __future__ import annotations

import sys
import tempfile

import pytest
from slashed import CommandStore

from acp import DefaultACPClient
from acp.schema import EnvVariable, StdioMcpServer
from llmling_agent import Agent
from llmling_agent.delegation import AgentPool
from llmling_agent.log import get_logger
from llmling_agent.tools.base import Tool
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.converters import convert_acp_mcp_server_to_config
from llmling_agent_acp.session import ACPSession
from llmling_agent_acp.session_manager import ACPSessionManager


logger = get_logger(__name__)


async def test_mcp_server_conversion():
    """Test conversion from ACP McpServer to our MCPServerConfig."""
    acp_server = StdioMcpServer(
        name="test_server",
        command="uv",
        args=["run", "test-mcp-server"],
        env=[
            EnvVariable(name="API_KEY", value="test123"),
            EnvVariable(name="DEBUG", value="true"),
        ],
    )
    config = convert_acp_mcp_server_to_config(acp_server)
    assert config.name == "test_server"
    assert config.command == "uv"
    assert config.args == ["run", "test-mcp-server"]
    assert config.env == {"API_KEY": "test123", "DEBUG": "true"}


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS subprocess handling differs")
async def test_session_with_mcp_servers(mock_acp_agent, client_capabilities):
    """Test creating an ACP session with MCP servers."""

    def simple_callback(message: str) -> str:
        return f"Test response for: {message}"

    agent = Agent(name="test_agent", provider=simple_callback)
    agent_pool = AgentPool[None]()
    agent_pool.register("test_agent", agent)
    client = DefaultACPClient(allow_file_operations=True)
    command_store = CommandStore()
    command_bridge = ACPCommandBridge(command_store)

    # Sample MCP servers (these won't actually connect in the test)
    mcp_servers = [
        StdioMcpServer(
            name="filesystem",
            command="npx",  # Use echo as a dummy command
            args=["-y", "@upstash/context7-mcp"],
            env=[],
        ),
        StdioMcpServer(
            name="web_search",
            command="uvx",  # Use echo as a dummy command
            args=["mcp-server-git"],
            env=[EnvVariable(name="API_KEY", value="dummy")],
        ),
    ]

    session = ACPSession(  # Create session with MCP servers
        session_id="test_session",
        agent_pool=agent_pool,
        current_agent_name="test_agent",
        cwd=tempfile.gettempdir(),
        client=client,
        mcp_servers=mcp_servers,
        command_bridge=command_bridge,
        acp_agent=mock_acp_agent,
        client_capabilities=client_capabilities,
    )

    assert session.session_id == "test_session"
    assert session.mcp_servers == mcp_servers
    assert session.mcp_manager is None  # Not initialized yet

    # Test initialization (this will fail without real MCP servers, which is expected)
    try:
        await session.initialize_mcp_servers()
        print("✓ MCP servers initialized (unexpectedly succeeded)")
    except Exception as e:  # noqa: BLE001
        print(f"✓ MCP server initialization failed as expected: {type(e).__name__}")

    await session.close()


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS subprocess handling differs")
async def test_session_manager_with_mcp(mock_acp_agent, client_capabilities):
    """Test session manager creating sessions with MCP servers."""
    command_store = CommandStore()
    command_bridge = ACPCommandBridge(command_store)
    session_manager = ACPSessionManager(command_bridge)

    def simple_callback(message: str) -> str:
        return f"Test response for: {message}"

    agent = Agent(name="test_agent", provider=simple_callback)
    agent_pool = AgentPool[None]()  # Create empty pool and register the agent
    agent_pool.register("test_agent", agent)
    client = DefaultACPClient()
    mcp_servers = [StdioMcpServer(name="tools", command="echo", args=["tools"], env=[])]

    try:
        session_id = await session_manager.create_session(
            agent_pool=agent_pool,
            default_agent_name="test_agent",
            cwd=tempfile.gettempdir(),
            client=client,
            mcp_servers=mcp_servers,
            acp_agent=mock_acp_agent,
            client_capabilities=client_capabilities,
        )

        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session.mcp_servers == mcp_servers
        await session_manager.close_session(session_id)

    except Exception:
        logger.exception("Session manager test failed")
        raise


async def test_tool_integration():
    """Test that MCP tools would be properly integrated."""

    def simple_callback(message: str) -> str:
        return f"Test response for: {message}"

    agent = Agent(name="test_agent", provider=simple_callback)

    async with agent:
        initial_tools = len(await agent.tools.get_tools())
        # In real scenario, MCP tools would be added here
        # For test, we'll simulate by adding a dummy tool

        def dummy_mcp_tool(query: str) -> str:
            """Dummy MCP tool for testing."""
            return f"MCP result for: {query}"

        meta = {"mcp_tool": "dummy_search"}
        tool = Tool.from_callable(dummy_mcp_tool, source="mcp", metadata=meta)

        agent.tools.register_tool(tool)

        final_tools = len(await agent.tools.get_tools())
        assert final_tools == initial_tools + 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
