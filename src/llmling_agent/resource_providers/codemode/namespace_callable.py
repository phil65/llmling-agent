"""Namespace callable wrapper for tools."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.utils.inspection import execute


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

    name_override: str | None = None
    """Override name for the callable, defaults to callable.__name__."""

    def __post_init__(self) -> None:
        """Set function attributes for introspection."""
        self.__name__ = self.name_override or self.callable.__name__
        self.__doc__ = self.callable.__doc__ or ""

    @property
    def name(self) -> str:
        """Get the effective name of the callable."""
        return self.name_override or self.callable.__name__

    @classmethod
    def from_tool(cls, tool: Tool) -> NamespaceCallable:
        """Create a NamespaceCallable from a Tool.

        Args:
            tool: The tool to wrap

        Returns:
            NamespaceCallable instance
        """
        name_override = tool.name if tool.name != tool.callable.__name__ else None
        return cls(tool.callable, name_override)

    @classmethod
    def from_generator(cls, generator: ToolCodeGenerator) -> NamespaceCallable:
        """Create a NamespaceCallable from a ToolCodeGenerator.

        Args:
            generator: The generator to wrap

        Returns:
            NamespaceCallable instance
        """
        return cls(generator.callable, generator.name_override)

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the wrapped callable."""
        try:
            result = await execute(self.callable, *args, **kwargs)
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
