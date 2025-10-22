"""Provider for tool management tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling import RuntimeConfig


def create_tool_management_tools(runtime: RuntimeConfig | None = None) -> list[Tool]:
    """Create tools for tool management operations."""
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


class ToolManagementTools(StaticResourceProvider):
    """Provider for tool management tools."""

    def __init__(
        self, name: str = "tool_management", runtime: RuntimeConfig | None = None
    ):
        super().__init__(name=name, tools=create_tool_management_tools(runtime))
