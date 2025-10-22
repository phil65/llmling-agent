"""Provider for agent management tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


def create_agent_management_tools() -> list[Tool]:
    """Create tools for agent and team management operations."""
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


class AgentManagementTools(StaticResourceProvider):
    """Provider for agent management tools."""

    def __init__(self, name: str = "agent_management"):
        super().__init__(name=name, tools=create_agent_management_tools())
