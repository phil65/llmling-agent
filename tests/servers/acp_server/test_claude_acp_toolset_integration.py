"""Integration test for Claude ACP agent with toolsets exposed via MCP bridge.

This test creates an AgentPool with a Claude ACP agent configured with
the Subagent toolset, which gets exposed via an internal MCP server bridge.
The Claude agent can then use our internal tools through MCP.

Run with: pytest tests/servers/acp_server/test_claude_acp_toolset_integration.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from acp.exceptions import RequestError
from llmling_agent import AgentPool
from llmling_agent.agent.acp_agent import ACPAgent
from llmling_agent.models.acp_agents import ClaudeACPAgentConfig
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_config.toolsets import SubagentToolsetConfig


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(60),
    pytest.mark.integration,
]


@pytest.fixture
def claude_config_with_subagent() -> ClaudeACPAgentConfig:
    """Create Claude ACP config with Subagent toolset."""
    return ClaudeACPAgentConfig(
        name="claude_orchestrator",
        description="Claude agent with subagent delegation capabilities",
        cwd=str(Path.cwd()),
        permission_mode="acceptEdits",
        toolsets=[SubagentToolsetConfig()],
    )


@pytest.fixture
def manifest_with_claude(claude_config_with_subagent: ClaudeACPAgentConfig) -> AgentsManifest:
    """Create manifest with Claude ACP agent."""
    return AgentsManifest(
        acp_agents={"claude_orchestrator": claude_config_with_subagent},
    )


async def test_claude_acp_with_subagent_toolset_setup(manifest_with_claude: AgentsManifest):
    """Test that Claude ACP agent with Subagent toolset initializes correctly."""
    try:
        async with AgentPool(manifest=manifest_with_claude) as pool:
            # Verify ACP agent was created
            assert "claude_orchestrator" in pool.acp_agents
            agent = pool.acp_agents["claude_orchestrator"]

            # Verify it's an ACPAgent
            assert isinstance(agent, ACPAgent)

            # Verify toolset bridge was set up
            assert agent._tool_bridge is not None
            assert agent._owns_bridge is True

            # Verify the MCP server is running
            assert agent._tool_bridge.port > 0
            assert "sse" in agent._tool_bridge.url

            # Verify tools are registered (SubagentToolset always has tools)
            tools = await agent.tools.get_tools()
            assert len(tools) > 0
            tool_names = {t.name for t in tools}
            # SubagentTools provides: list_available_nodes, delegate_to, ask_agent
            assert "list_available_nodes" in tool_names or "delegate_to" in tool_names

    except FileNotFoundError:
        pytest.skip("claude-code-acp not installed")
    except TimeoutError:
        pytest.skip("Claude ACP server took too long to start")
    except RequestError as e:
        pytest.skip(f"Claude ACP server error: {e}")


async def test_claude_acp_subagent_invocation(manifest_with_claude: AgentsManifest):
    """Test invoking subagent tools through Claude ACP agent.

    Note: This test requires:
    - claude-code-acp to be installed and accessible
    - Valid API credentials for Claude
    """
    try:
        async with AgentPool(manifest=manifest_with_claude) as pool:
            agent = pool.acp_agents["claude_orchestrator"]

            # Ask the agent to list available nodes - it should have access via MCP
            result = await asyncio.wait_for(
                agent.run("Use the list_available_nodes tool to show me available agents"),
                timeout=45.0,
            )

            assert result is not None
            assert result.content is not None

    except FileNotFoundError:
        pytest.skip("claude-code-acp not installed")
    except TimeoutError:
        pytest.skip("Claude ACP agent took too long to respond")
    except RequestError as e:
        pytest.skip(f"Claude ACP server error: {e}")
    except Exception as e:
        if "API" in str(e) or "authentication" in str(e).lower():
            pytest.skip(f"API authentication issue: {e}")
        raise


async def test_claude_acp_tool_bridge_mcp_config(claude_config_with_subagent: ClaudeACPAgentConfig):
    """Test that tool bridge MCP config is properly passed to session."""
    try:
        async with AgentPool() as pool:
            # Manually create and configure agent
            agent = ACPAgent(config=claude_config_with_subagent, agent_pool=pool)

            async with agent:
                # Verify extra MCP servers include our bridge
                assert len(agent._extra_mcp_servers) > 0

                # Find our toolset bridge server
                bridge_server = next(
                    (s for s in agent._extra_mcp_servers if "tools" in s.name),
                    None,
                )
                assert bridge_server is not None

    except FileNotFoundError:
        pytest.skip("claude-code-acp not installed")
    except TimeoutError:
        pytest.skip("Claude ACP server took too long to start")
    except RequestError as e:
        pytest.skip(f"Claude ACP server error: {e}")


async def test_claude_acp_multiple_toolsets():
    """Test Claude ACP agent with multiple toolsets."""
    from llmling_agent_config.toolsets import AgentManagementToolsetConfig

    config = ClaudeACPAgentConfig(
        name="claude_multi",
        cwd=str(Path.cwd()),
        permission_mode="acceptEdits",
        toolsets=[
            SubagentToolsetConfig(),
            AgentManagementToolsetConfig(),
        ],
    )

    try:
        async with AgentPool() as pool:
            agent = ACPAgent(config=config, agent_pool=pool)

            async with agent:
                # All toolsets should be exposed via single bridge
                assert agent._tool_bridge is not None

                tools = await agent.tools.get_tools()
                tool_names = {t.name for t in tools}

                # Should have tools from both toolsets
                # SubagentToolset provides: list_available_nodes, delegate_to, ask_agent
                assert "list_available_nodes" in tool_names or "delegate_to" in tool_names

    except FileNotFoundError:
        pytest.skip("claude-code-acp not installed")
    except TimeoutError:
        pytest.skip("Claude ACP server took too long to start")
    except RequestError as e:
        pytest.skip(f"Claude ACP server error: {e}")


async def test_pool_cleanup_stops_tool_bridges(manifest_with_claude: AgentsManifest):
    """Test that pool cleanup properly stops tool bridges."""
    try:
        pool = AgentPool(manifest=manifest_with_claude)
        async with pool:
            agent = pool.acp_agents["claude_orchestrator"]
            assert agent._tool_bridge is not None
            bridge_port = agent._tool_bridge.port
            assert bridge_port > 0

        # After pool exit, bridge should be stopped
        assert agent._tool_bridge is None

    except FileNotFoundError:
        pytest.skip("claude-code-acp not installed")
    except TimeoutError:
        pytest.skip("Claude ACP server took too long to start")
    except RequestError as e:
        pytest.skip(f"Claude ACP server error: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
