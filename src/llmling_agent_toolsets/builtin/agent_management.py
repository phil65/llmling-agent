"""Provider for agent management tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_agent_management_tools


class AgentManagementTools(StaticResourceProvider):
    """Provider for agent management tools."""

    def __init__(self, name: str = "agent_management"):
        super().__init__(name=name, tools=create_agent_management_tools())
