"""Tests for process management functionality."""

from __future__ import annotations


async def test_builtin_toolset_tools():
    """Test that process management tools are available in builtin toolset."""
    from llmling_agent_toolsets.builtin import ProcessManagementTools

    provider = ProcessManagementTools()
    tools = await provider.get_tools()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "start_process",
        "get_process_output",
        "wait_for_process",
        "kill_process",
        "release_process",
        "list_processes",
    ]

    for tool_name in expected_tools:
        assert tool_name in tool_names
