"""Aggregating resource provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling import BasePrompt
    from pydantic_ai import ModelRequestPart

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo


class AggregatingResourceProvider(ResourceProvider):
    """Provider that combines resources from multiple providers."""

    def __init__(self, providers: Sequence[ResourceProvider], name: str = "aggregating"):
        """Initialize provider with list of providers to aggregate.

        Args:
            providers: Resource providers to aggregate
            name: Name for this provider
        """
        super().__init__(name=name)
        self.providers = list(providers)

    async def get_tools(self) -> list[Tool]:
        """Get tools from all providers."""
        tools: list[Tool] = []
        for provider in self.providers:
            tools.extend(await provider.get_tools())
        return tools

    async def get_prompts(self) -> list[BasePrompt]:
        """Get prompts from all providers."""
        prompts: list[BasePrompt] = []
        for provider in self.providers:
            prompts.extend(await provider.get_prompts())
        return prompts

    async def get_resources(self) -> list[ResourceInfo]:
        """Get resources from all providers."""
        resources: list[ResourceInfo] = []
        for provider in self.providers:
            resources.extend(await provider.get_resources())
        return resources

    async def get_request_parts(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[ModelRequestPart]:
        """Try to get prompt from first provider that has it."""
        for provider in self.providers:
            try:
                return await provider.get_request_parts(name, arguments)
            except KeyError:
                continue
        msg = f"Prompt {name!r} not found in any provider"
        raise KeyError(msg)
