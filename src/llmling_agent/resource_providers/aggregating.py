"""Aggregating resource provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from pydantic_ai import ModelRequestPart

    from llmling_agent.prompts.prompts import BasePrompt
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo


class AggregatingResourceProvider(ResourceProvider):
    """Provider that combines resources from multiple providers."""

    def __init__(self, providers: list[ResourceProvider], name: str = "aggregating") -> None:
        """Initialize provider with list of providers to aggregate.

        Args:
            providers: Resource providers to aggregate (stores reference to list)
            name: Name for this provider
        """
        super().__init__(name=name)
        # Store reference to the providers list for dynamic updates
        self.providers = providers

    async def get_tools(self) -> list[Tool]:
        """Get tools from all providers."""
        return [t for provider in self.providers for t in await provider.get_tools()]

    async def get_prompts(self) -> list[BasePrompt]:
        """Get prompts from all providers."""
        return [p for provider in self.providers for p in await provider.get_prompts()]

    async def get_resources(self) -> list[ResourceInfo]:
        """Get resources from all providers."""
        return [r for provider in self.providers for r in await provider.get_resources()]

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
