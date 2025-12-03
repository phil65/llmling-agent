"""Entry point based toolset implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from epregistry import EntryPointRegistry

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class EntryPointTools(ResourceProvider):
    """Provider for entry point based tools."""

    def __init__(self, module: str) -> None:
        super().__init__(name=module)
        self.module = module
        self._tools: list[Tool] | None = None
        self.registry = EntryPointRegistry[Callable[..., Any]]("llmling")

    async def get_tools(self) -> list[Tool]:
        """Get tools from entry points."""
        # Return cached tools if available
        if self._tools is not None:
            return self._tools

        self._tools = []
        entry_point = self.registry.get("tools")
        if not entry_point:
            msg = f"No tools entry point found for {self.module}"
            raise ValueError(msg)

        get_tools = entry_point.load()
        for item in get_tools():
            meta = {"module": self.module}
            tool = self.create_tool(item, metadata=meta)
            self._tools.append(tool)
        return self._tools
