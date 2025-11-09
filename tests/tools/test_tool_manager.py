"""Tests for tool management."""

from __future__ import annotations

import pytest

from llmling_agent.tools import Tool, ToolError, ToolManager


async def test_basic_tool_management():
    """Test basic tool enabling/disabling."""
    tool1 = Tool.from_callable(lambda x: x, name_override="tool1")
    tool2 = Tool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])

    await manager.disable_tool("tool1")
    t_1 = await manager.get_tool("tool1")
    t_2 = await manager.get_tool("tool2")
    assert t_1
    assert t_2
    assert not t_1.enabled
    assert t_2.enabled


async def test_priority_sorting():
    """Test tools are sorted by priority."""
    tool1 = Tool.from_callable(lambda x: x, name_override="tool1")
    tool2 = Tool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    tool = await manager.get_tool("tool1")
    assert tool
    tool.priority = 200
    tool = await manager.get_tool("tool2")
    assert tool
    tool.priority = 100

    tools = await manager.get_tools()
    assert [t.name for t in tools] == ["tool2", "tool1"]


async def test_state_filtering():
    """Test filtering tools by state."""
    tool1 = Tool.from_callable(lambda x: x, name_override="tool1")
    tool2 = Tool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    await manager.disable_tool("tool1")

    enabled = await manager.get_tools(state="enabled")
    assert len(enabled) == 1
    assert enabled[0].name == "tool2"

    disabled = await manager.get_tools(state="disabled")
    assert len(disabled) == 1
    assert disabled[0].name == "tool1"


async def test_invalid_tool_operations():
    """Test error handling for invalid tool operations."""
    manager = ToolManager()

    with pytest.raises(ToolError):
        await manager.enable_tool("nonexistent")

    with pytest.raises(ToolError):
        await manager.disable_tool("nonexistent")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
