"""Provider for tool management tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_tool_management_tools


if TYPE_CHECKING:
    from llmling import RuntimeConfig


class ToolManagementTools(StaticResourceProvider):
    """Provider for tool management tools."""

    def __init__(
        self, name: str = "tool_management", runtime: RuntimeConfig | None = None
    ):
        super().__init__(name=name, tools=create_tool_management_tools(runtime))
