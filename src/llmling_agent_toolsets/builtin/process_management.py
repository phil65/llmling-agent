"""Provider for process management tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


def create_process_management_tools() -> list[Tool]:
    """Create tools for process management operations."""
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


class ProcessManagementTools(StaticResourceProvider):
    """Provider for process management tools."""

    def __init__(self, name: str = "process_management"):
        super().__init__(name=name, tools=create_process_management_tools())
