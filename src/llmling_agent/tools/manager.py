"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, assert_never

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent.utils.baseregistry import LLMLingError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.common_types import ToolType
    from llmling_agent.prompts.prompts import MCPClientPrompt
    from llmling_agent.resource_providers import ResourceProvider
    from llmling_agent.resource_providers.codemode.provider import CodeModeResourceProvider


logger = get_logger(__name__)

MAX_LEN_DESCRIPTION = 2000
ToolState = Literal["all", "enabled", "disabled"]
ProviderName = str
OwnerType = Literal["pool", "team", "node"]
ToolMode = Literal["codemode"]


class ToolError(LLMLingError):
    """Base exception for tool-related errors."""


class ToolManager:
    """Manages tool registration, enabling/disabling and access."""

    def __init__(
        self,
        tools: Sequence[ToolType] | None = None,
        tool_mode: ToolMode | None = None,
    ) -> None:
        """Initialize tool manager.

        Args:
            tools: Initial tools to register
            tool_mode: Tool execution mode (None or "codemode")
        """
        super().__init__()
        self.external_providers: list[ResourceProvider] = []
        self.worker_provider = StaticResourceProvider(name="workers")
        self.builtin_provider = StaticResourceProvider(name="builtin")
        self.tool_mode = tool_mode

        # CodeModeResourceProvider gets populated with providers in providers property
        from llmling_agent.resource_providers.codemode.provider import CodeModeResourceProvider

        self._codemode_provider: CodeModeResourceProvider = CodeModeResourceProvider([])

        # Forward to provider methods
        self.tool = self.builtin_provider.tool
        self.register_tool = self.builtin_provider.register_tool
        self.register_worker = self.worker_provider.register_worker

        # Register initial tools
        for tool in tools or []:
            t = self._validate_item(tool)
            self.builtin_provider.add_tool(t)

    @property
    def providers(self) -> list[ResourceProvider]:
        """Get all providers: external + worker + builtin providers."""
        if self.tool_mode == "codemode":
            # Update the providers list with current providers
            self._codemode_provider.providers[:] = [
                *self.external_providers,
                self.worker_provider,
                self.builtin_provider,
            ]
            return [self._codemode_provider]

        return [*self.external_providers, self.worker_provider, self.builtin_provider]

    async def __prompt__(self) -> str:
        enabled_tools = [t.name for t in await self.get_tools() if t.enabled]
        if not enabled_tools:
            return "No tools available"
        return f"Available tools: {', '.join(enabled_tools)}"

    def add_provider(self, provider: ResourceProvider, owner: str | None = None) -> None:
        """Add an external resource provider.

        Args:
            provider: ResourceProvider instance (e.g., MCP server, custom provider)
            owner: Optional owner for the provider
        """
        if owner:
            provider.owner = owner
        self.external_providers.append(provider)

    def remove_provider(self, provider: ResourceProvider | ProviderName) -> None:
        """Remove an external resource provider."""
        from llmling_agent.resource_providers import ResourceProvider

        match provider:
            case ResourceProvider():
                self.external_providers.remove(provider)
            case str():
                for p in self.external_providers:
                    if p.name == provider:
                        self.external_providers.remove(p)
            case _ as unreachable:
                assert_never(unreachable)

    async def reset_states(self) -> None:
        """Reset all tools to their default enabled states."""
        for info in await self.get_tools():
            info.enabled = True

    def _validate_item(self, item: ToolType) -> Tool:
        """Validate and convert items before registration."""
        match item:
            case Tool():
                return item
            case Callable() | str():
                return Tool.from_callable(item)
            case _:
                typ = type(item)
                msg = f"Item must be Tool or callable. Got {typ}"
                raise ToolError(msg)

    async def enable_tool(self, tool_name: str) -> None:
        """Enable a previously disabled tool."""
        tool_info = await self.get_tool(tool_name)
        tool_info.enabled = True
        logger.debug("Enabled tool", tool_name=tool_name)

    async def disable_tool(self, tool_name: str) -> None:
        """Disable a tool."""
        tool_info = await self.get_tool(tool_name)
        tool_info.enabled = False
        logger.debug("Disabled tool", tool_name=tool_name)

    async def list_tools(self) -> dict[str, bool]:
        """Get a mapping of all tools and their enabled status."""
        return {tool.name: tool.enabled for tool in await self.get_tools()}

    async def get_tools(
        self,
        state: ToolState = "all",
        names: str | list[str] | None = None,
    ) -> list[Tool]:
        """Get tool objects based on filters."""
        tools: list[Tool] = []
        # Get tools from providers concurrently
        provider_coroutines = [provider.get_tools() for provider in self.providers]
        results = await asyncio.gather(*provider_coroutines, return_exceptions=True)
        for provider, result in zip(self.providers, results, strict=False):
            if isinstance(result, BaseException):
                msg = "Failed to get tools from provider"
                logger.warning(msg, provider=provider, result=result)
                continue
            tools.extend(t for t in result if t.matches_filter(state))

        match names:
            case str():
                tools = [t for t in tools if t.name == names]
            case list():
                tools = [t for t in tools if t.name in names]
        return tools

    async def get_tool(self, name: str) -> Tool:
        """Get a specific tool by name.

        First checks local tools, then uses concurrent provider fetching.
        """
        tool = next((tool for tool in await self.get_tools() if tool.name == name), None)
        if not tool:
            msg = f"Tool not found: {tool}"
            raise ToolError(msg)
        return tool

    async def list_prompts(self) -> list[MCPClientPrompt]:
        """Get all prompts from all providers."""
        from llmling_agent.mcp_server.manager import MCPManager

        all_prompts: list[MCPClientPrompt] = []
        # Get prompts from all external providers (check if they're MCP providers)
        for provider in self.external_providers:
            if isinstance(provider, MCPManager):
                try:
                    # Get prompts from MCP providers via the aggregating provider
                    agg_provider = provider.get_aggregating_provider()
                    prompts = await agg_provider.get_prompts()
                    all_prompts.extend(prompts)
                except Exception:
                    logger.exception("Failed to get prompts from provider", provider=provider)

        return all_prompts

    @asynccontextmanager
    async def temporary_tools(
        self,
        tools: ToolType | Sequence[ToolType],
        *,
        exclusive: bool = False,
    ) -> AsyncIterator[list[Tool]]:
        """Temporarily register tools.

        Args:
            tools: Tool(s) to register
            exclusive: Whether to temporarily disable all other tools

        Yields:
            List of registered tool infos
        """
        # Normalize inputs to lists
        tools_list: list[ToolType] = [tools] if not isinstance(tools, Sequence) else list(tools)
        # Store original tool states if exclusive
        tools = await self.get_tools()
        original_states: dict[str, bool] = {}
        if exclusive:
            original_states = {t.name: t.enabled for t in tools}
            # Disable all existing tools
            for t in tools:
                t.enabled = False
        # Register all tools
        registered_tools: list[Tool] = []
        try:
            for tool in tools_list:
                tool_info = self.register_tool(tool)
                registered_tools.append(tool_info)
            yield registered_tools

        finally:
            # Remove temporary tools
            for tool_info in registered_tools:
                self.builtin_provider.remove_tool(tool_info.name)
            # Restore original tool states if exclusive
            if exclusive:
                for name_, was_enabled in original_states.items():
                    t_ = await self.get_tool(name_)
                    t_.enabled = was_enabled


if __name__ == "__main__":
    manager = ToolManager()

    @manager.tool(name="custom_name", description="Custom description")
    def with_params(query: str) -> str:
        """With parameters."""
        return query.upper()
