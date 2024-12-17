"""Tests for tool management."""

from __future__ import annotations

from llmling.tools import LLMCallableTool
import pytest

from llmling_agent.tools.manager import ToolError, ToolManager


def test_basic_tool_management() -> None:
    """Test basic tool enabling/disabling."""
    tool1 = LLMCallableTool.from_callable(lambda x: x, name_override="tool1")
    tool2 = LLMCallableTool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    assert manager.is_tool_enabled("tool1")

    manager.disable_tool("tool1")
    assert not manager.is_tool_enabled("tool1")
    assert manager.is_tool_enabled("tool2")

    # Test enabling again
    manager.enable_tool("tool1")
    assert manager.is_tool_enabled("tool1")


def test_priority_sorting() -> None:
    """Test tools are sorted by priority."""
    tool1 = LLMCallableTool.from_callable(lambda x: x, name_override="tool1")
    tool2 = LLMCallableTool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    manager["tool1"].priority = 200
    manager["tool2"].priority = 100

    tools = manager.get_tools()
    assert [t.name for t in tools] == ["tool2", "tool1"]


def test_state_filtering() -> None:
    """Test filtering tools by state."""
    tool1 = LLMCallableTool.from_callable(lambda x: x, name_override="tool1")
    tool2 = LLMCallableTool.from_callable(lambda x: x, name_override="tool2")

    manager = ToolManager([tool1, tool2])
    manager.disable_tool("tool1")

    enabled = manager.get_tools(state="enabled")
    assert len(enabled) == 1
    assert enabled[0].name == "tool2"

    disabled = manager.get_tools(state="disabled")
    assert len(disabled) == 1
    assert disabled[0].name == "tool1"


def test_invalid_tool_operations() -> None:
    """Test error handling for invalid tool operations."""
    manager = ToolManager()

    with pytest.raises(ToolError):
        manager.enable_tool("nonexistent")

    with pytest.raises(ToolError):
        manager.disable_tool("nonexistent")


if __name__ == "__main__":
    pytest.main(["-v", __file__])