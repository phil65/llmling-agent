"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent.utils.baseregistry import LLMLingError
from llmling_agent.utils.importing import import_class


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent import Agent, MessageNode
    from llmling_agent.common_types import AnyCallable, ToolSource, ToolType
    from llmling_agent.prompts.prompts import Prompt
    from llmling_agent.resource_providers import ResourceProvider


logger = get_logger(__name__)

MAX_LEN_DESCRIPTION = 2000
ToolState = Literal["all", "enabled", "disabled"]
ProviderName = str
OwnerType = Literal["pool", "team", "node"]


class ToolError(LLMLingError):
    """Base exception for tool-related errors."""


class ToolManager:
    """Manages tool registration, enabling/disabling and access.

    Inherits from BaseRegistry providing:
    - Dict-like access: manager["tool_name"] -> Tool
    - Async startup/shutdown: await manager.startup()
    - Event observation: manager.add_observer(observer)
    - Registration: manager.register("tool_name", tool)
    - Listing: manager.list_items()
    - State check: manager.is_empty, manager.has_item()
    - Async iteration: async for name, tool in manager: ...
    """

    def __init__(
        self,
        tools: Sequence[Tool | ToolType] | None = None,
    ):
        """Initialize tool manager.

        Args:
            tools: Initial tools to register
        """
        super().__init__()
        self.external_providers: list[ResourceProvider] = []
        self.worker_provider = StaticResourceProvider(name="workers")
        self.builtin_provider = StaticResourceProvider(name="builtin")
        # Register initial tools
        for tool in tools or []:
            t = self._validate_item(tool)
            self.builtin_provider.add_tool(t)

    @property
    def providers(self) -> list[ResourceProvider]:
        """Get all providers: external + worker + builtin providers."""
        return [*self.external_providers, self.worker_provider, self.builtin_provider]

    async def __prompt__(self) -> str:
        enabled_tools = [t.name for t in await self.get_tools() if t.enabled]
        if not enabled_tools:
            return "No tools available"
        return f"Available tools: {', '.join(enabled_tools)}"

    def add_provider(
        self,
        provider: ResourceProvider,
        owner: str | None = None,
    ):
        """Add an external resource provider.

        Args:
            provider: ResourceProvider instance (e.g., MCP server, custom provider)
            owner: Optional owner for the provider
        """
        if owner:
            provider.owner = owner
        self.external_providers.append(provider)

    def remove_provider(self, provider: ResourceProvider | ProviderName):
        """Remove an external resource provider."""
        from llmling_agent.resource_providers import ResourceProvider

        match provider:
            case ResourceProvider():
                self.external_providers.remove(provider)
            case str():
                for p in self.external_providers:
                    if p.name == provider:
                        self.external_providers.remove(p)
            case _:
                msg = f"Invalid provider type: {type(provider)}"
                raise ValueError(msg)

    async def reset_states(self) -> None:
        """Reset all tools to their default enabled states."""
        for info in await self.get_tools():
            info.enabled = True

    def _validate_item(self, item: Tool | ToolType) -> Tool:
        """Validate and convert items before registration."""
        match item:
            case Tool():
                return item
            case str():
                if item.startswith("crewai_tools"):
                    obj = import_class(item)()
                    return Tool.from_crewai_tool(obj)
                if item.startswith("langchain"):
                    obj = import_class(item)()
                    return Tool.from_langchain_tool(obj)
                return Tool.from_callable(item)
            case Callable():
                return Tool.from_callable(item)
            case _:
                typ = type(item)
                msg = f"Item must be Tool or callable. Got {typ}"
                raise ToolError(msg)

    async def enable_tool(self, tool_name: str):
        """Enable a previously disabled tool."""
        tool_info = await self.get_tool(tool_name)
        if not tool_info:
            msg = f"Tool not found: {tool_name}"
            raise ToolError(msg)
        tool_info.enabled = True
        logger.debug("Enabled tool", tool_name=tool_name)

    async def disable_tool(self, tool_name: str):
        """Disable a tool."""
        tool_info = await self.get_tool(tool_name)
        if not tool_info:
            msg = f"Tool not found: {tool_name}"
            raise ToolError(msg)
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
                logger.exception("Failed to get tools from provider", provider=provider)
                continue
            tools.extend(t for t in result if t.matches_filter(state))

        # Sort by priority if any have non-default priority
        match names:
            case str():
                tools = [t for t in tools if t.name == names]
            case list():
                tools = [t for t in tools if t.name in names]
        if any(t.priority != 100 for t in tools):  # noqa: PLR2004
            tools.sort(key=lambda t: t.priority)
        return tools

    async def get_tool(self, name: str) -> Tool | None:
        """Get a specific tool by name.

        First checks local tools, then uses concurrent provider fetching.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool instance if found, None otherwise
        """
        all_tools = await self.get_tools()
        return next((tool for tool in all_tools if tool.name == name), None)

    async def get_tool_names(self, state: ToolState = "all") -> set[str]:
        """Get tool names based on state."""
        return {t.name for t in await self.get_tools() if t.matches_filter(state)}

    async def list_prompts(self) -> list[Prompt]:
        """Get all prompts from all providers.

        Returns:
            List of Prompt instances
        """
        from llmling_agent.mcp_server.manager import MCPManager

        all_prompts: list[Prompt] = []

        # Get prompts from all external providers (check if they're MCP providers)
        for provider in self.external_providers:
            if isinstance(provider, MCPManager):
                try:
                    prompts = await provider.list_prompts()
                    all_prompts.extend(prompts)
                except Exception:
                    logger.exception(
                        "Failed to get prompts from provider", provider=provider
                    )

        return all_prompts

    def register_tool(
        self,
        tool: ToolType | Tool,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        enabled: bool = True,
        source: ToolSource = "dynamic",
        priority: int = 100,
        requires_confirmation: bool = False,
        requires_capability: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Tool:
        """Register a new tool with custom settings.

        Args:
            tool: Tool to register (callable, or import path)
            enabled: Whether tool is initially enabled
            name_override: Optional name override for the tool
            description_override: Optional description override for the tool
            source: Tool source (runtime/agent/builtin/dynamic)
            priority: Execution priority (lower = higher priority)
            requires_confirmation: Whether tool needs confirmation
            requires_capability: Optional capability needed to use tool
            metadata: Additional tool metadata

        Returns:
            Created Tool instance
        """
        # First convert to basic Tool
        match tool:
            case Tool():
                tool.description = description_override or tool.description
                tool.name = name_override or tool.name
                tool.source = source
                tool.metadata = tool.metadata | (metadata or {})

            case _:
                tool = Tool.from_callable(
                    tool,
                    enabled=enabled,
                    source=source,
                    name_override=name_override,
                    description_override=description_override,
                    priority=priority,
                    requires_confirmation=requires_confirmation,
                    requires_capability=requires_capability,
                    metadata=metadata or {},
                )

        # Register the tool
        self.builtin_provider.add_tool(tool)
        return tool

    def register_worker(
        self,
        worker: MessageNode[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        parent: Agent[Any, Any] | None = None,
    ) -> Tool:
        """Register an agent as a worker tool.

        Args:
            worker: Agent to register as worker
            name: Optional name override for the worker tool
            reset_history_on_run: Whether to clear history before each run
            pass_message_history: Whether to pass parent's message history
            parent: Optional parent agent for history/context sharing
        """
        from llmling_agent import Agent, BaseTeam

        match worker:
            case BaseTeam():
                tool = worker.to_tool(name=name)
            case Agent():
                tool = worker.to_tool(
                    parent=parent,
                    name=name,
                    reset_history_on_run=reset_history_on_run,
                    pass_message_history=pass_message_history,
                )
            case _:
                msg = f"Unsupported worker type: {type(worker)}"
                raise ValueError(msg)
        msg = "Registering worker as tool"
        logger.debug(msg, worker_name=worker.name, tool_name=tool.name)
        self.worker_provider.add_tool(tool)
        return tool

    @asynccontextmanager
    async def temporary_tools(
        self,
        tools: ToolType | Tool | Sequence[ToolType | Tool],
        *,
        exclusive: bool = False,
    ) -> AsyncIterator[list[Tool]]:
        """Temporarily register tools.

        Args:
            tools: Tool(s) to register
            exclusive: Whether to temporarily disable all other tools

        Yields:
            List of registered tool infos

        Example:
            ```python
            with tool_manager.temporary_tools([tool1, tool2], exclusive=True) as tools:
                # Only tool1 and tool2 are available
                await agent.run(prompt)
            # Original tool states are restored
            ```
        """
        # Normalize inputs to lists
        tools_list: list[ToolType | Tool] = (
            [tools] if not isinstance(tools, Sequence) else list(tools)
        )

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
                    if t_ := await self.get_tool(name_):
                        t_.enabled = was_enabled

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        enabled: bool = True,
        source: ToolSource = "dynamic",
        priority: int = 100,
        requires_confirmation: bool = False,
        requires_capability: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Callable[[AnyCallable], AnyCallable]:
        """Decorator to register a function as a tool.

        Args:
            name: Optional override for tool name (defaults to function name)
            description: Optional description override
            enabled: Whether tool is initially enabled
            source: Tool source type
            priority: Execution priority (lower = higher)
            requires_confirmation: Whether tool needs confirmation
            requires_capability: Optional required capability
            metadata: Additional tool metadata

        Returns:
            Decorator function that registers the tool

        Example:
            @tool_manager.register(
                name="search_docs",
                description="Search documentation",
                requires_confirmation=True
            )
            async def search(query: str) -> str:
                '''Search the docs.'''
                return "Results..."
        """

        def decorator(func: AnyCallable) -> AnyCallable:
            self.register_tool(
                func,
                name_override=name,
                description_override=description,
                enabled=enabled,
                source=source,
                priority=priority,
                requires_confirmation=requires_confirmation,
                requires_capability=requires_capability,
                metadata=metadata,
            )
            return func

        return decorator
