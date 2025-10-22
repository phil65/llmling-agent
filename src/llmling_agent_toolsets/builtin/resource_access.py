"""Provider for resource access tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_resource_access_tools


if TYPE_CHECKING:
    from llmling import RuntimeConfig


class ResourceAccessTools(StaticResourceProvider):
    """Provider for resource access tools."""

    def __init__(
        self, name: str = "resource_access", runtime: RuntimeConfig | None = None
    ):
        super().__init__(name=name, tools=create_resource_access_tools(runtime))
