"""Provider for RuntimeConfig tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.log import get_logger
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent_config.resources import ResourceInfo


if TYPE_CHECKING:
    from llmling import RuntimeConfig
    from llmling.prompts import BasePrompt

logger = get_logger(__name__)


class RuntimeResourceProvider(ResourceProvider):
    """Provider that exposes RuntimeConfig tools through ResourceProvider interface."""

    def __init__(self, runtime: RuntimeConfig, name: str = "runtime"):
        """Initialize provider with RuntimeConfig.

        Args:
            runtime: RuntimeConfig instance to wrap
            name: Name of the provider
        """
        super().__init__(name=name)
        self._runtime = runtime

    async def get_tools(self) -> list[Tool]:
        """Convert RuntimeConfig tools to Tools."""
        tools: list[Tool] = []

        for tool in self._runtime.get_tools():
            try:
                tools.append(Tool(tool, source="runtime"))
            except Exception:
                logger.exception("Failed to convert runtime tool", name=tool.name)
                continue

        return tools

    async def get_prompts(self) -> list[BasePrompt]:
        """Get prompts from runtime."""
        return list(self._runtime.get_prompts())

    async def get_resources(self) -> list[ResourceInfo]:
        """Convert runtime resources to ResourceInfo."""
        resources: list[ResourceInfo] = []

        for resource in self._runtime.get_resources():
            try:
                name = resource.name or str(resource.uri)
                uri = str(resource.uri)
                info = ResourceInfo(name=name, uri=uri, description=resource.description)
                resources.append(info)
            except Exception:
                logger.exception("Failed to convert runtime resource", name=resource.name)
                continue

        return resources
