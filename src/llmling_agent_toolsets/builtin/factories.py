"""Factory functions for creating builtin tool collections."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from llmling import RuntimeConfig

    from llmling_agent.tools.base import Tool


def create_agent_management_tools() -> list[Tool]:
    """Create tools for agent and team management operations."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.delegate_to,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.list_available_agents,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.list_available_teams,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.create_worker_agent,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.spawn_delegate,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.add_agent,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.add_team,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.ask_agent,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.connect_nodes,
            source="builtin",
            category="other",
        ),
    ]


def create_file_access_tools() -> list[Tool]:
    """Create tools for file and directory access operations."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.read_file,
            source="builtin",
            category="read",
        ),
        Tool.from_callable(
            capability_tools.list_directory,
            source="builtin",
            category="search",
        ),
    ]


def create_code_execution_tools() -> list[Tool]:
    """Create tools for code execution operations."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.execute_python,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.execute_command,
            source="builtin",
            category="execute",
        ),
    ]


def create_process_management_tools() -> list[Tool]:
    """Create tools for process management operations."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.start_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.get_process_output,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.wait_for_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.kill_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.release_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.list_processes,
            source="builtin",
            category="search",
        ),
    ]


def create_resource_access_tools(runtime: RuntimeConfig | None = None) -> list[Tool]:
    """Create tools for resource access operations."""
    from llmling_agent.tools.base import Tool

    tools: list[Tool] = []

    # Resource tools require runtime
    if runtime:
        tools.extend([
            Tool.from_callable(
                runtime.load_resource,
                source="builtin",
                category="read",
            ),
            Tool.from_callable(
                runtime.get_resources,
                source="builtin",
                category="search",
            ),
        ])

    return tools


def create_tool_management_tools(runtime: RuntimeConfig | None = None) -> list[Tool]:
    """Create tools for tool management operations."""
    from llmling_agent.tools.base import Tool

    tools: list[Tool] = []

    # Tool management requires runtime
    if runtime:
        tools.extend([
            Tool.from_callable(
                runtime.register_tool,
                source="builtin",
                category="other",
            ),
            Tool.from_callable(
                runtime.register_code_tool,
                source="builtin",
                category="other",
            ),
            Tool.from_callable(
                runtime.install_package,
                source="builtin",
                category="execute",
            ),
        ])

    return tools


def create_user_interaction_tools() -> list[Tool]:
    """Create tools for user interaction operations."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.ask_user,
            source="builtin",
            category="other",
        ),
    ]


def create_history_tools() -> list[Tool]:
    """Create tools for history and statistics access."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.search_history,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.show_statistics,
            source="builtin",
            category="read",
        ),
    ]


def create_code_tools() -> list[Tool]:
    """Create tools for code formatting and linting."""
    from llmling_agent.tools.base import Tool
    from llmling_agent_toolsets.code import format_code

    return [
        Tool.from_callable(
            format_code,
            source="builtin",
            category="execute",
        ),
    ]
