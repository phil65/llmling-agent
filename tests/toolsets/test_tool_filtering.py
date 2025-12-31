"""Tests for tool filtering in toolset configurations."""

from __future__ import annotations

from agentpool_config.toolsets import (
    CodeToolsetConfig,
    SkillsToolsetConfig,
    SubagentToolsetConfig,
    UserInteractionToolsetConfig,
)


async def test_subagent_tool_filtering():
    """Test filtering tools in subagent toolset."""
    config = SubagentToolsetConfig(tools={"delegate_to": True, "ask_agent": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "delegate_to" in tool_names
    assert "list_available_nodes" in tool_names  # not in filter, defaults to True
    assert "ask_agent" not in tool_names


async def test_user_interaction_tool_filtering():
    """Test filtering tools in user interaction toolset."""
    config = UserInteractionToolsetConfig(tools={"ask_user": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "ask_user" not in tool_names
    assert len(tool_names) == 0


async def test_skills_tool_filtering():
    """Test filtering tools in skills toolset."""
    config = SkillsToolsetConfig(tools={"load_skill": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "load_skill" not in tool_names
    assert "list_skills" in tool_names


async def test_code_toolset_filtering():
    """Test filtering tools in code toolset."""
    config = CodeToolsetConfig(tools={"format_code": True, "ast_grep": False})
    provider = config.get_provider()
    tools = await provider.get_tools()
    tool_names = {t.name for t in tools}

    assert "format_code" in tool_names
    assert "ast_grep" not in tool_names


async def test_filtering_provider_delegates_attributes():
    """Test that FilteringResourceProvider delegates attributes correctly."""
    config = SubagentToolsetConfig(tools={"delegate_to": True})
    provider = config.get_provider()

    # Should delegate name attribute
    assert provider.name == "subagent_tools"
    # Should have log attribute from ResourceProvider
    assert hasattr(provider, "log")
