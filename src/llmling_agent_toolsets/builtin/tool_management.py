"""Provider for tool management tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Callable


async def register_tool(  # noqa: D417
    ctx: AgentContext,
    tool: str | Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    enabled: bool = True,
) -> str:
    """Register a new tool from callable or import path.

    Args:
        tool: Callable function or import path string to register as tool
        name: Optional name override for the tool
        description: Optional description override for the tool
        enabled: Whether the tool should be enabled initially

    Returns:
        Confirmation message with registered tool name
    """
    # Create tool from callable/import path
    tool_obj = Tool.from_callable(
        tool,
        name_override=name,
        description_override=description,
        source="dynamic",
        enabled=enabled,
    )

    # Register with the agent's tool manager
    registered_tool = ctx.agent.tools.register_tool(tool_obj)

    return f"Successfully registered tool: {registered_tool.name}"


async def register_code_tool(  # noqa: D417
    ctx: AgentContext,
    code: str,
    name: str | None = None,
    description: str | None = None,
    enabled: bool = True,
) -> str:
    """Register a new tool from code string.

    Args:
        code: Python code string containing a callable function
        name: Optional name override for the tool
        description: Optional description override for the tool
        enabled: Whether the tool should be enabled initially

    Returns:
        Confirmation message with registered tool name
    """
    # Create tool from code
    tool_obj = Tool.from_code(
        code,
        name=name,
        description=description,
    )
    tool_obj.enabled = enabled
    tool_obj.source = "dynamic"

    # Register with the agent's tool manager
    registered_tool = ctx.agent.tools.register_tool(tool_obj)

    return f"Successfully registered code tool: {registered_tool.name}"


class ToolManagementTools(StaticResourceProvider):
    """Provider for tool management tools."""

    def __init__(self, name: str = "tool_management") -> None:
        super().__init__(name=name)
        self._tools = [
            self.create_tool(register_tool, category="other"),
            self.create_tool(register_code_tool, category="other"),
        ]
