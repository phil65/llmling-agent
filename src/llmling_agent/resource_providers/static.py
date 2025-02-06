"""Static resource provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.base import ResourceProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.prompts import BasePrompt

    from llmling_agent.models.resources import ResourceInfo
    from llmling_agent.tools.base import ToolInfo


class StaticResourceProvider(ResourceProvider):
    """Provider for pre-configured tools, prompts and resources.

    Allows creating a provider that serves a fixed set of resources
    passed during initialization. Useful for converting static configurations
    to the common ResourceProvider interface.
    """

    def __init__(
        self,
        tools: Sequence[ToolInfo] | None = None,
        prompts: Sequence[BasePrompt] | None = None,
        resources: Sequence[ResourceInfo] | None = None,
    ) -> None:
        """Initialize provider with static resources.

        Args:
            tools: Optional list of tools to serve
            prompts: Optional list of prompts to serve
            resources: Optional list of resources to serve
        """
        self._tools = list(tools) if tools else []
        self._prompts = list(prompts) if prompts else []
        self._resources = list(resources) if resources else []

    async def get_tools(self) -> list[ToolInfo]:
        """Get pre-configured tools."""
        return self._tools

    async def get_prompts(self) -> list[BasePrompt]:
        """Get pre-configured prompts."""
        return self._prompts

    async def get_resources(self) -> list[ResourceInfo]:
        """Get pre-configured resources."""
        return self._resources
