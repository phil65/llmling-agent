"""Orchestrates code generation for multiple tools."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers.codemode.tool_code_generator import (
    ToolCodeGenerator,
)


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool


class NamespaceCallable:
    """Wrapper for tool functions with proper repr and call interface."""

    def __init__(
        self,
        tool: Tool,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
    ) -> None:
        """Initialize tool callable wrapper.

        Args:
            tool: The tool to wrap
            name_override: Override the tool name
            description_override: Override the tool description
        """
        self._tool = tool
        self._name_override = name_override
        self._description_override = description_override
        self.__name__ = name_override or tool.name
        self.__doc__ = description_override or tool.description

    @classmethod
    def from_tool(cls, tool: Tool) -> NamespaceCallable:
        """Create a NamespaceCallable from a Tool.

        Args:
            tool: The tool to wrap

        Returns:
            NamespaceCallable instance
        """
        return cls(tool)

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the wrapped tool."""
        try:
            result = await self._tool.execute(*args, **kwargs)
            # Handle coroutines that weren't properly awaited
            if inspect.iscoroutine(result):
                result = await result
            # Ensure we return a serializable value
        except Exception as e:  # noqa: BLE001
            return f"Error executing {self._tool.name}: {e!s}"
        else:
            return result if result is not None else "Operation completed successfully"

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"NamespaceCallable(name='{self._tool.name}', description='{self._tool.description[:50]}...')"  # noqa: E501

    def __str__(self) -> str:
        """Return readable string representation."""
        return f"<tool: {self._tool.name}>"

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._name_override or self._tool.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self._description_override or self._tool.description

    @property
    def signature(self) -> str:
        """Get function signature for debugging."""
        generator = ToolCodeGenerator.from_tool(self._tool)
        return generator.get_function_signature()
