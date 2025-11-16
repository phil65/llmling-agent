"""Static resource provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from llmling_agent.prompts.prompts import BasePrompt
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo


class StaticResourceProvider(ResourceProvider):
    """Provider for pre-configured tools, prompts and resources.

    Allows creating a provider that serves a fixed set of resources
    passed during initialization. Useful for converting static configurations
    to the common ResourceProvider interface.
    """

    def __init__(
        self,
        name: str = "static",
        tools: Sequence[Tool] | None = None,
        prompts: Sequence[BasePrompt] | None = None,
        resources: Sequence[ResourceInfo] | None = None,
    ) -> None:
        """Initialize provider with static resources.

        Args:
            name: Name of the provider
            tools: Optional list of tools to serve
            prompts: Optional list of prompts to serve
            resources: Optional list of resources to serve
        """
        super().__init__(name=name)
        self._tools = list(tools) if tools else []
        self._prompts = list(prompts) if prompts else []
        self._resources = list(resources) if resources else []

    async def get_tools(self) -> list[Tool]:
        """Get pre-configured tools."""
        return self._tools

    async def get_tool(self, name: str) -> Tool | None:
        """Get tool by name."""
        return next((tool for tool in self._tools if tool.name == name), None)

    async def get_prompts(self) -> list[BasePrompt]:
        """Get pre-configured prompts."""
        return self._prompts

    async def get_resources(self) -> list[ResourceInfo]:
        """Get pre-configured resources."""
        return self._resources

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to this provider.

        Args:
            tool: Tool to add
        """
        self._tools.append(tool)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name.

        Args:
            name: Name of tool to remove

        Returns:
            True if tool was found and removed, False otherwise
        """
        for i, tool in enumerate(self._tools):
            if tool.name == name:
                self._tools.pop(i)
                return True
        return False

    def add_prompt(self, prompt: BasePrompt) -> None:
        """Add a prompt to this provider.

        Args:
            prompt: Prompt to add
        """
        self._prompts.append(prompt)

    def remove_prompt(self, name: str) -> bool:
        """Remove a prompt by name.

        Args:
            name: Name of prompt to remove

        Returns:
            True if prompt was found and removed, False otherwise
        """
        for i, prompt in enumerate(self._prompts):
            if prompt.name == name:
                self._prompts.pop(i)
                return True
        return False

    def add_resource(self, resource: ResourceInfo) -> None:
        """Add a resource to this provider.

        Args:
            resource: Resource to add
        """
        self._resources.append(resource)

    def remove_resource(self, name: str) -> bool:
        """Remove a resource by name.

        Args:
            name: Name of resource to remove

        Returns:
            True if resource was found and removed, False otherwise
        """
        for i, resource in enumerate(self._resources):
            if resource.name == name:
                self._resources.pop(i)
                return True
        return False

    @overload
    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]: ...

    @overload
    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        requires_confirmation: bool = False,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        requires_confirmation: bool = False,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool.

        Can be used with or without parameters:

        ```python
        # Without parameters
        @provider.tool
        def my_function(x: int) -> str:
            return str(x)

        # With parameters
        @provider.tool(name="custom_name", description="Custom description")
        def another_function(y: str) -> str:
            return y.upper()
        ```

        Args:
            func: Function to register (when used without parentheses)
            name: Override for tool name
            description: Override for tool description
            enabled: Whether tool is initially enabled
            requires_confirmation: Whether execution needs confirmation
            metadata: Additional tool metadata
            **kwargs: Additional arguments passed to Tool.from_callable
        """
        from llmling_agent.tools.base import Tool

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool = Tool.from_callable(
                f,
                name_override=name,
                description_override=description,
                enabled=enabled,
                requires_confirmation=requires_confirmation,
                metadata=metadata or {},
                **kwargs,
            )
            self.add_tool(tool)
            return f

        if func is None:
            # Called with arguments: @provider.tool(...)
            return decorator
        # Called without arguments: @provider.tool
        return decorator(func)
