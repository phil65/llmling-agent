"""Tests for tool filtering in toolset configurations."""

from __future__ import annotations

import pytest

from agentpool_config.toolsets import (
    AgentManagementToolsetConfig,
    CodeToolsetConfig,
    ExecutionEnvironmentToolsetConfig,
    HistoryToolsetConfig,
    IntegrationToolsetConfig,
    SkillsToolsetConfig,
    SubagentToolsetConfig,
    ToolManagementToolsetConfig,
    UserInteractionToolsetConfig,
)


@pytest.mark.asyncio
async def test_agent_management_tool_filtering():
    """Test filtering tools in agent management toolset."""
    config = AgentManagementToolsetConfig(
        tools={
            "create_worker_agent": True,
            "add_agent": False,
            "add_team": True,
            "connect_nodes": False,
        }
    )
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "create_worker_agent" in tool_names
    assert "add_team" in tool_names
    assert "add_agent" not in tool_names
    assert "connect_nodes" not in tool_names


@pytest.mark.asyncio
async def test_agent_management_no_filter():
    """Test that all tools are enabled when no filter is specified."""
    config = AgentManagementToolsetConfig()
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "create_worker_agent" in tool_names
    assert "add_agent" in tool_names
    assert "add_team" in tool_names
    assert "connect_nodes" in tool_names


@pytest.mark.asyncio
async def test_subagent_tool_filtering():
    """Test filtering tools in subagent toolset."""
    config = SubagentToolsetConfig(tools={"delegate_to": True, "ask_agent": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "delegate_to" in tool_names
    assert "list_available_nodes" in tool_names  # not in filter, defaults to True
    assert "ask_agent" not in tool_names


@pytest.mark.asyncio
async def test_user_interaction_tool_filtering():
    """Test filtering tools in user interaction toolset."""
    config = UserInteractionToolsetConfig(tools={"ask_user": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "ask_user" not in tool_names
    assert len(tool_names) == 0


@pytest.mark.asyncio
async def test_history_tool_filtering():
    """Test filtering tools in history toolset."""
    config = HistoryToolsetConfig(tools={"search_history": True, "show_statistics": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "search_history" in tool_names
    assert "show_statistics" not in tool_names


@pytest.mark.asyncio
async def test_skills_tool_filtering():
    """Test filtering tools in skills toolset."""
    config = SkillsToolsetConfig(tools={"load_skill": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "load_skill" not in tool_names
    assert "list_skills" in tool_names


@pytest.mark.asyncio
async def test_integration_tool_filtering():
    """Test filtering tools in integration toolset."""
    config = IntegrationToolsetConfig(
        tools={"add_local_mcp_server": True, "add_remote_mcp_server": False}
    )
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "add_local_mcp_server" in tool_names
    assert "add_remote_mcp_server" not in tool_names


@pytest.mark.asyncio
async def test_tool_management_filtering():
    """Test filtering tools in tool management toolset."""
    config = ToolManagementToolsetConfig(tools={"register_tool": True, "register_code_tool": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "register_tool" in tool_names
    assert "register_code_tool" not in tool_names


@pytest.mark.asyncio
async def test_execution_environment_filtering():
    """Test filtering tools in execution environment toolset."""
    config = ExecutionEnvironmentToolsetConfig(
        tools={
            "execute_code": True,
            "execute_command": False,
            "start_process": True,
            "kill_process": False,
        }
    )
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "execute_code" in tool_names
    assert "start_process" in tool_names
    assert "execute_command" not in tool_names
    assert "kill_process" not in tool_names
    # Tools not in filter default to True
    assert "get_process_output" in tool_names


@pytest.mark.asyncio
async def test_code_toolset_filtering():
    """Test filtering tools in code toolset."""
    config = CodeToolsetConfig(tools={"format_code": True, "ast_grep": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "format_code" in tool_names
    assert "ast_grep" not in tool_names


@pytest.mark.asyncio
async def test_filtering_provider_delegates_attributes():
    """Test that FilteringResourceProvider delegates attributes correctly."""
    config = AgentManagementToolsetConfig(tools={"create_worker_agent": True})
    provider = config.get_provider()

    # Should delegate name attribute
    assert provider.name == "agent_management"
    # Should have log attribute from ResourceProvider
    assert hasattr(provider, "log")
