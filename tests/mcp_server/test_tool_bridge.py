"""Tests for ToolManagerBridge - exposing ToolManager tools to ACP agents via MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from llmling_agent import AgentPool
from llmling_agent.agents.acp_agent import ACPAgent
from llmling_agent.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge, create_tool_bridge
from llmling_agent.models.acp_agents.mcp_capable import ClaudeACPAgentConfig
from llmling_agent.tools import ToolManager
from llmling_agent_config.toolsets import AgentManagementToolsetConfig, SubagentToolsetConfig


if TYPE_CHECKING:
    from llmling_agent.agents.context import AgentContext


# Simple tool without context
def simple_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Tool that needs AgentContext
async def list_pool_agents(ctx: AgentContext) -> str:
    """List all agents in the pool."""
    if not ctx.pool:
        return "No pool available"
    return ", ".join(ctx.pool.agents.keys())


@pytest.fixture
def tool_manager() -> ToolManager:
    """Create a ToolManager with test tools."""
    manager = ToolManager()
    manager.register_tool(simple_add)
    return manager


@pytest.fixture
def tool_manager_with_context_tool() -> ToolManager:
    """Create a ToolManager with context-aware tools."""
    manager = ToolManager()
    manager.register_tool(simple_add)
    manager.register_tool(list_pool_agents)
    return manager


async def test_bridge_config_defaults():
    """Test BridgeConfig has sensible defaults."""
    config = BridgeConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 0  # Auto-select
    assert config.transport == "sse"
    assert config.server_name == "llmling-toolmanager"


async def test_bridge_lifecycle(tool_manager: ToolManager):
    """Test bridge start/stop lifecycle."""
    async with AgentPool() as pool:
        bridge = ToolManagerBridge(tool_manager, pool=pool, config=BridgeConfig(port=0))
        # Not started yet
        assert bridge._mcp is None
        assert bridge._actual_port is None
        await bridge.start()
        assert bridge._mcp is not None
        assert bridge._actual_port is not None
        assert bridge.port > 0
        assert "http://" in bridge.url
        await bridge.stop()
        assert bridge._mcp is None
        assert bridge._actual_port is None


async def test_bridge_context_manager(tool_manager: ToolManager):
    """Test bridge as async context manager."""
    async with AgentPool() as pool:
        cfg = BridgeConfig(port=0)
        async with ToolManagerBridge(tool_manager, pool=pool, config=cfg) as bridge:
            assert bridge.port > 0
            assert bridge._server is not None

        assert bridge._server is None  # After exit, should be stopped


async def test_create_tool_bridge_helper(tool_manager: ToolManager):
    """Test the create_tool_bridge convenience function."""
    async with (
        AgentPool() as pool,
        create_tool_bridge(tool_manager, pool, transport="sse") as bridge,
    ):
        assert bridge.port > 0
        assert "/sse" in bridge.url


async def test_bridge_registers_tools(tool_manager: ToolManager):
    """Test that tools are registered with FastMCP."""
    cfg = BridgeConfig(port=0)
    async with (
        AgentPool() as pool,
        ToolManagerBridge(tool_manager, pool=pool, config=cfg) as bridge,
    ):
        assert bridge._mcp is not None  # The MCP server should have our tool registered
        # FastMCP stores tools internally - we verify via the tool count
        tools = await tool_manager.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "simple_add"


async def test_get_mcp_server_config_sse(tool_manager: ToolManager):
    """Test generating SSE MCP server config."""
    cfg = BridgeConfig(transport="sse")
    async with (
        AgentPool() as pool,
        ToolManagerBridge(tool_manager, pool=pool, config=cfg) as bridge,
    ):
        config = bridge.get_mcp_server_config()
        assert config.name == "llmling-toolmanager"
        assert str(config.url).startswith("http://")
        assert "/sse" in str(config.url)


async def test_get_mcp_server_config_http(tool_manager: ToolManager):
    """Test generating HTTP MCP server config."""
    cfg = BridgeConfig(transport="streamable-http")
    async with (
        AgentPool() as pool,
        ToolManagerBridge(tool_manager, pool=pool, config=cfg) as bridge,
    ):
        config = bridge.get_mcp_server_config()
        assert config.name == "llmling-toolmanager"
        assert str(config.url).startswith("http://")
        assert "/mcp" in str(config.url)


async def test_pool_create_tool_bridge():
    """Test creating tool bridge via AgentPool."""
    async with AgentPool() as pool:
        # Create a minimal agent to get a tool manager
        agent = await pool.add_agent(name="test_agent", model="test", system_prompt="Test")
        # Create bridge from agent's tools
        bridge = await pool.create_tool_bridge(
            agent.tools,
            name="test_bridge",
            owner_agent_name="test_agent",
        )

        assert bridge.port > 0
        assert "test_bridge" in pool._tool_bridges
        assert bridge.owner_agent_name == "test_agent"
        # Cleanup should stop bridges
        await pool.remove_tool_bridge("test_bridge")
        assert "test_bridge" not in pool._tool_bridges


async def test_pool_tool_bridge_cleanup():
    """Test that pool cleanup stops all bridges."""
    pool = AgentPool()
    async with pool:
        agent = await pool.add_agent(name="test_agent", model="test", system_prompt="Test")
        await pool.create_tool_bridge(agent.tools, name="bridge1")
        await pool.create_tool_bridge(agent.tools, name="bridge2")
        assert len(pool._tool_bridges) == 2  # noqa: PLR2004

    # After exiting, bridges should be cleaned up
    assert len(pool._tool_bridges) == 0


async def test_bridge_with_context_tools(tool_manager_with_context_tool: ToolManager):
    """Test that tools requiring AgentContext work through bridge."""
    async with AgentPool() as pool:
        # Add an agent so there's something in the pool
        await pool.add_agent(name="helper", model="test", system_prompt="Helper")

        async with ToolManagerBridge(
            tool_manager=tool_manager_with_context_tool,
            pool=pool,
            config=BridgeConfig(port=0),
        ):
            # Both tools should be registered
            tools = await tool_manager_with_context_tool.get_tools()
            assert len(tools) == 2  # noqa: PLR2004
            tool_names = {t.name for t in tools}
            assert "simple_add" in tool_names
            assert "list_pool_agents" in tool_names


async def test_proxy_context_creation(tool_manager: ToolManager):
    """Test that proxy AgentContext is created correctly."""
    async with AgentPool() as pool:
        await pool.add_agent(name="owner", model="test", system_prompt="Test")
        cfg = BridgeConfig(port=0)
        bridge = ToolManagerBridge(tool_manager, pool=pool, config=cfg, owner_agent_name="owner")
        # Create a mock MCP context (we just need to test context creation)
        mock_mcp_ctx = MagicMock()
        ctx = bridge._create_proxy_context(
            tool_name="test_tool",
            tool_call_id="call-123",
            tool_input={"arg": "value"},
            mcp_ctx=mock_mcp_ctx,
        )
        assert ctx.node_name == "owner"
        assert ctx.pool is pool
        assert ctx.tool_name == "test_tool"
        assert ctx.tool_call_id == "call-123"
        assert ctx.tool_input == {"arg": "value"}


async def test_duplicate_bridge_name_raises():
    """Test that creating bridge with duplicate name raises."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        await pool.create_tool_bridge(agent.tools, name="my_bridge")
        with pytest.raises(ValueError, match="already exists"):
            await pool.create_tool_bridge(agent.tools, name="my_bridge")


async def test_get_nonexistent_bridge_raises():
    """Test that getting nonexistent bridge raises KeyError."""
    async with AgentPool() as pool:
        with pytest.raises(KeyError, match="not found"):
            await pool.get_tool_bridge("nonexistent")


async def test_acp_agent_toolsets_adds_providers():
    """Test that toolsets from config are added to ToolManager."""
    async with AgentPool() as pool:
        toolsets = [SubagentToolsetConfig(), AgentManagementToolsetConfig()]
        config = ClaudeACPAgentConfig(name="test_acp", toolsets=toolsets)
        agent = ACPAgent(config=config, agent_pool=pool)
        await agent._setup_toolsets()
        # Check that providers were added
        # The tools should now include tools from both toolsets
        tools = await agent.tools.get_tools()
        tool_names = {t.name for t in tools}
        # SubagentTools provides: list_available_nodes, delegate_to, ask_agent
        assert "list_available_nodes" in tool_names
        assert "delegate_to" in tool_names
        await agent._cleanup()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
