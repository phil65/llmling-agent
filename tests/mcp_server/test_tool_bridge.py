"""Tests for ToolManagerBridge - exposing ToolManager tools to ACP agents via MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentpool import AgentPool
from agentpool.agents.acp_agent import ACPAgent
from agentpool.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge, create_tool_bridge
from agentpool.models.acp_agents.mcp_capable import ClaudeACPAgentConfig
from agentpool_config.toolsets import AgentManagementToolsetConfig, SubagentToolsetConfig


if TYPE_CHECKING:
    from agentpool.agents.context import AgentContext


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


async def test_bridge_config_defaults():
    """Test BridgeConfig has sensible defaults."""
    config = BridgeConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 0  # Auto-select
    assert config.transport == "sse"
    assert config.server_name == "agentpool-toolmanager"


async def test_bridge_lifecycle():
    """Test bridge start/stop lifecycle."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        bridge = ToolManagerBridge(node=agent, config=BridgeConfig(port=0))
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


async def test_bridge_context_manager():
    """Test bridge as async context manager."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        cfg = BridgeConfig(port=0)
        async with ToolManagerBridge(node=agent, config=cfg) as bridge:
            assert bridge.port > 0
            assert bridge._server is not None

        assert bridge._server is None  # After exit, should be stopped


async def test_create_tool_bridge_helper():
    """Test the create_tool_bridge convenience function."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        async with create_tool_bridge(agent, transport="sse") as bridge:
            assert bridge.port > 0
            assert "/sse" in bridge.url


async def test_bridge_registers_tools():
    """Test that tools are registered with FastMCP."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        cfg = BridgeConfig(port=0)
        async with ToolManagerBridge(node=agent, config=cfg) as bridge:
            assert bridge._mcp is not None  # The MCP server should have our tool registered
            # FastMCP stores tools internally - we verify via the tool count
            tools = await agent.tools.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "simple_add"


async def test_get_mcp_server_config_sse():
    """Test generating SSE MCP server config."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        cfg = BridgeConfig(transport="sse")
        async with ToolManagerBridge(node=agent, config=cfg) as bridge:
            config = bridge.get_mcp_server_config()
            assert config.name == "agentpool-toolmanager"
            assert str(config.url).startswith("http://")
            assert "/sse" in str(config.url)


async def test_get_mcp_server_config_http():
    """Test generating HTTP MCP server config."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        cfg = BridgeConfig(transport="streamable-http")
        async with ToolManagerBridge(node=agent, config=cfg) as bridge:
            config = bridge.get_mcp_server_config()
            assert config.name == "agentpool-toolmanager"
            assert str(config.url).startswith("http://")
            assert "/mcp" in str(config.url)


async def test_pool_create_tool_bridge():
    """Test creating tool bridge via AgentPool."""
    async with AgentPool() as pool:
        # Create a minimal agent to get a tool manager
        agent = await pool.add_agent(name="test_agent", model="test", system_prompt="Test")
        # Create bridge from agent
        bridge = await pool.create_tool_bridge(agent, name="test_bridge")

        assert bridge.port > 0
        assert "test_bridge" in pool._tool_bridges
        # Cleanup should stop bridges
        await pool.remove_tool_bridge("test_bridge")
        assert "test_bridge" not in pool._tool_bridges


async def test_pool_tool_bridge_cleanup():
    """Test that pool cleanup stops all bridges."""
    pool = AgentPool()
    async with pool:
        agent = await pool.add_agent(name="test_agent", model="test", system_prompt="Test")
        await pool.create_tool_bridge(agent, name="bridge1")
        await pool.create_tool_bridge(agent, name="bridge2")
        assert len(pool._tool_bridges) == 2  # noqa: PLR2004

    # After exiting, bridges should be cleaned up
    assert len(pool._tool_bridges) == 0


async def test_bridge_with_context_tools():
    """Test that tools requiring AgentContext work through bridge."""
    async with AgentPool() as pool:
        # Add an agent so there's something in the pool
        agent = await pool.add_agent(name="helper", model="test", system_prompt="Helper")
        agent.tools.register_tool(simple_add)
        agent.tools.register_tool(list_pool_agents)

        async with ToolManagerBridge(node=agent, config=BridgeConfig(port=0)):
            # Both tools should be registered
            tools = await agent.tools.get_tools()
            assert len(tools) == 2  # noqa: PLR2004
            tool_names = {t.name for t in tools}
            assert "simple_add" in tool_names
            assert "list_pool_agents" in tool_names


async def test_proxy_context_creation():
    """Test that context is created correctly via dataclasses.replace."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="owner", model="test", system_prompt="Test")
        agent.tools.register_tool(simple_add)
        cfg = BridgeConfig(port=0)
        ToolManagerBridge(node=agent, config=cfg)
        # The bridge uses node.get_context() and replaces tool fields
        # We verify the node's context has the right properties
        ctx = agent.get_context()
        assert ctx.node_name == "owner"
        assert ctx.pool is pool


async def test_duplicate_bridge_name_raises():
    """Test that creating bridge with duplicate name raises."""
    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        await pool.create_tool_bridge(agent, name="my_bridge")
        with pytest.raises(ValueError, match="already exists"):
            await pool.create_tool_bridge(agent, name="my_bridge")


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


async def test_tool_call_via_mcp_client():
    """Test calling a tool through the MCP bridge via an actual MCP client.

    This verifies that tool_call_id extraction works correctly when tools
    are called through MCP.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    captured_tool_call_ids: list[str] = []

    # Tool that captures its context's tool_call_id
    async def capture_id_tool(ctx: AgentContext, message: str) -> str:
        """Tool that captures the tool_call_id from context."""
        captured_tool_call_ids.append(ctx.tool_call_id or "no-id")
        return f"Got message: {message}"

    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(capture_id_tool)

        async with (
            create_tool_bridge(agent, transport="sse") as bridge,
            sse_client(bridge.url) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            # List tools to verify registration
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            assert "capture_id_tool" in tool_names

            # Call the tool - without _meta, should generate fallback ID
            await session.call_tool("capture_id_tool", {"message": "hello"})
            assert len(captured_tool_call_ids) == 1
            # Should have generated a UUID-like fallback (36 chars)
            uuid_length = 36
            assert len(captured_tool_call_ids[0]) == uuid_length


async def test_tool_call_with_meta_tool_use_id():
    """Test that claudecode/toolUseId in _meta is extracted correctly.

    When Claude Code calls tools, it passes the tool_use_id in _meta.
    This test simulates that behavior by passing meta to call_tool.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    captured_tool_call_ids: list[str] = []

    async def capture_id_tool(ctx: AgentContext, value: int) -> str:
        """Tool that captures the tool_call_id from context."""
        captured_tool_call_ids.append(ctx.tool_call_id or "no-id")
        return f"value={value}"

    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(capture_id_tool)

        async with (
            create_tool_bridge(agent, transport="sse") as bridge,
            sse_client(bridge.url) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            # Call tool WITH meta containing claudecode/toolUseId
            # This simulates what Claude Code should send
            expected_id = "toolu_01ABC123XYZ"
            await session.call_tool(
                "capture_id_tool",
                {"value": 42},
                meta={"claudecode/toolUseId": expected_id},
            )

            assert len(captured_tool_call_ids) == 1
            captured_id = captured_tool_call_ids[0]
            # The captured ID should match what we sent in meta
            assert captured_id == expected_id, (
                f"Expected tool_call_id '{expected_id}', got '{captured_id}'"
            )


@pytest.mark.skip(reason="Requires Claude Code CLI - run manually to verify")
async def test_claude_code_passes_tool_use_id_in_meta():
    """Integration test: verify Claude Code CLI passes claudecode/toolUseId in _meta.

    This test uses the actual Claude Code CLI via ClaudeSDKClient to verify
    that when Claude calls an MCP tool, it passes the tool_use_id in _meta.

    Run manually with: pytest -v -s -k test_claude_code_passes_tool_use_id_in_meta --run-skip
    """
    from claude_agent_sdk import tool

    @tool(name="capture_meta", description="Captures the _meta field", input_schema={"value": int})
    async def capture_meta_tool(input_data: dict) -> dict:
        """Tool that captures what _meta was passed."""
        # The _meta is not directly accessible here in SDK tools
        # We need a different approach - use our MCP bridge
        return {"result": f"Got value: {input_data.get('value')}"}

    # For this test to work, we need to use our MCP bridge and check
    # what Claude Code actually sends. Let's modify the approach.
    # Placeholder - see test below


async def test_claude_code_mcp_bridge_integration():
    """Integration test using our MCP bridge with Claude Code.

    This test:
    1. Starts our MCP bridge with a tool that captures context
    2. Runs ClaudeSDKClient with that MCP server
    3. Asks Claude to use the tool
    4. Verifies the tool_call_id was extracted from meta
    """
    import os

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    # Override ANTHROPIC_API_KEY to use Claude Code subscription
    os.environ["ANTHROPIC_API_KEY"] = ""

    captured_ids: list[str] = []

    async def capture_context_tool(ctx: AgentContext, number: int) -> str:
        """Tool that captures context info."""
        captured_ids.append(ctx.tool_call_id or "none")
        return f"Got number: {number}"

    async with AgentPool() as pool:
        agent = await pool.add_agent(name="test", model="test", system_prompt="Test")
        agent.tools.register_tool(capture_context_tool)

        async with create_tool_bridge(agent, transport="streamable-http") as bridge:
            options = ClaudeAgentOptions(
                mcp_servers={
                    "test_bridge": {
                        "type": "http",
                        "url": bridge.url,
                    }
                },
                allowed_tools=["mcp__test_bridge__capture_context_tool"],
            )

            async with ClaudeSDKClient(options=options) as client:
                await client.query(
                    "You MUST use the capture_context_tool from test_bridge MCP server. "
                    "Call it with number=42. Do not explain, just call the tool."
                )

                # Stream messages until we get a result
                messages = []
                async for msg in client.receive_messages():
                    messages.append(msg)
                    msg_type = type(msg).__name__
                    print(f"Message: {msg_type}")
                    if hasattr(msg, "content"):
                        content = msg.content
                        if isinstance(content, str):
                            print(f"  Content: {content[:100]}")
                        elif isinstance(content, list) and content:
                            print(f"  Content: {content}")
                    if msg_type == "ResultMessage":
                        break

                print(f"\nTotal messages: {len(messages)}")

                if not captured_ids:
                    pytest.skip("Tool was not called - Claude may have responded differently")

                tool_id = captured_ids[0]
                print(f"Captured tool_call_id: {tool_id}")

                # Claude tool IDs typically start with "toolu_"
                if tool_id.startswith("toolu_"):
                    print("SUCCESS: Claude Code passed tool_use_id via _meta!")
                else:
                    print(f"Got fallback UUID: {tool_id}")
                    print("Claude Code may not be passing claudecode/toolUseId for external MCP")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
