"""Provider for resource access tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling import RuntimeConfig


def create_resource_access_tools(runtime: RuntimeConfig | None = None) -> list[Tool]:
    """Create tools for resource access operations."""
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


class ResourceAccessTools(StaticResourceProvider):
    """Provider for resource access tools."""

    def __init__(
        self, name: str = "resource_access", runtime: RuntimeConfig | None = None
    ):
        super().__init__(name=name, tools=create_resource_access_tools(runtime))
