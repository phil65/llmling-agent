"""Tests for tool management."""

from __future__ import annotations

import pytest

from llmling_agent.tools import Tool, ToolError, ToolManager


def test_basic_tool_management():
    """Test basic tool enabling/disabling."""
    tool1 = Tool.from_callable(lambda x: x, name_override="tool1")
    tool2 = Tool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    assert manager.is_tool_enabled("tool1")

    manager.disable_tool("tool1")
    assert not manager.is_tool_enabled("tool1")
    assert manager.is_tool_enabled("tool2")

    # Test enabling again
    manager.enable_tool("tool1")
    assert manager.is_tool_enabled("tool1")


async def test_priority_sorting():
    """Test tools are sorted by priority."""
    tool1 = Tool.from_callable(lambda x: x, name_override="tool1")
    tool2 = Tool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    manager["tool1"].priority = 200
    manager["tool2"].priority = 100

    tools = await manager.get_tools()
    assert [t.name for t in tools] == ["tool2", "tool1"]


async def test_state_filtering():
    """Test filtering tools by state."""
    tool1 = Tool.from_callable(lambda x: x, name_override="tool1")
    tool2 = Tool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    manager.disable_tool("tool1")

    enabled = await manager.get_tools(state="enabled")
    assert len(enabled) == 1
    assert enabled[0].name == "tool2"

    disabled = await manager.get_tools(state="disabled")
    assert len(disabled) == 1
    assert disabled[0].name == "tool1"


def test_invalid_tool_operations():
    """Test error handling for invalid tool operations."""
    manager = ToolManager()

    with pytest.raises(ToolError):
        manager.enable_tool("nonexistent")

    with pytest.raises(ToolError):
        manager.disable_tool("nonexistent")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
