"""Provider for process management tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_process_management_tools


class ProcessManagementTools(StaticResourceProvider):
    """Provider for process management tools."""

    def __init__(self, name: str = "process_management"):
        super().__init__(name=name, tools=create_process_management_tools())
