"""Namespace callable wrapper for tools."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent.resource_providers.codemode.tool_code_generator import (
        ToolCodeGenerator,
    )
    from llmling_agent.tools.base import Tool


@dataclass
class NamespaceCallable:
    """Wrapper for tool functions with proper repr and call interface."""

    callable: Callable
    """The callable function to execute."""

    name: str
    """Name of the callable."""

    def __post_init__(self) -> None:
        """Set function attributes for introspection."""
        self.__name__ = self.name
        self.__doc__ = self.callable.__doc__ or ""

    @classmethod
    def from_tool(cls, tool: Tool) -> NamespaceCallable:
        """Create a NamespaceCallable from a Tool.

        Args:
            tool: The tool to wrap

        Returns:
            NamespaceCallable instance
        """
        return cls(tool.callable, tool.name)

    @classmethod
    def from_generator(cls, generator: ToolCodeGenerator) -> NamespaceCallable:
        """Create a NamespaceCallable from a ToolCodeGenerator.

        Args:
            generator: The generator to wrap

        Returns:
            NamespaceCallable instance
        """
        return cls(generator.callable, generator.name)

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the wrapped callable."""
        try:
            # Check if callable is async
            if inspect.iscoroutinefunction(self.callable):
                result = await self.callable(*args, **kwargs)
            else:
                result = self.callable(*args, **kwargs)

            # Handle coroutines that weren't properly awaited
            if inspect.iscoroutine(result):
                result = await result
            # Ensure we return a serializable value
        except Exception as e:  # noqa: BLE001
            return f"Error executing {self.name}: {e!s}"
        else:
            return result if result is not None else "Operation completed successfully"

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"NamespaceCallable(name='{self.name}')"

    def __str__(self) -> str:
        """Return readable string representation."""
        return f"<tool: {self.name}>"

    @property
    def signature(self) -> str:
        """Get function signature for debugging."""
        try:
            sig = inspect.signature(self.callable)
        except (ValueError, TypeError):
            return f"{self.name}(...)"
        else:
            return f"{self.name}{sig}"
