"""Resource provider wrapper for callable functions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.utils.inspection import execute


if TYPE_CHECKING:
    from llmling import BasePrompt

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo


T = TypeVar("T")
ResourceCallable = Callable[[], T | Awaitable[T]]


class CallableResourceProvider(ResourceProvider):
    """Wraps callables as a resource provider.

    Handles both sync and async functions transparently.
    """

    def __init__(
        self,
        name: str = "callable",
        tool_callable: ResourceCallable[list[Tool]] | None = None,
        prompt_callable: ResourceCallable[list[BasePrompt]] | None = None,
        resource_callable: ResourceCallable[list[ResourceInfo]] | None = None,
    ):
        """Initialize provider with optional callables.

        Args:
            name: Name of the provider
            tool_callable: Function providing tools
            prompt_callable: Function providing prompts
            resource_callable: Function providing resources

        Each callable can be sync or async.
        """
        super().__init__(name=name)
        self.tool_callable = tool_callable
        self.prompt_callable = prompt_callable
        self.resource_callable = resource_callable

    async def _call_provider[T](
        self,
        provider: ResourceCallable[T] | None,
        default: T,
    ) -> T:
        """Helper to handle sync/async provider calls."""
        if not provider:
            return default
        return await execute(provider)

    async def get_tools(self) -> list[Tool]:
        """Get tools from callable if provided."""
        return await self._call_provider(self.tool_callable, [])

    async def get_prompts(self) -> list[BasePrompt]:
        """Get prompts from callable if provided."""
        return await self._call_provider(self.prompt_callable, [])

    async def get_resources(self) -> list[ResourceInfo]:
        """Get resources from callable if provided."""
        return await self._call_provider(self.resource_callable, [])
