"""User interaction tool for asking clarifying questions."""

from __future__ import annotations

from typing import Literal

from agentpool.tool_impls.ask_user.tool import AskUserTool
from agentpool_config.tools import ToolHints


__all__ = ["AskUserTool", "create_ask_user_tool"]

# Tool metadata defaults
NAME = "ask_user"
DESCRIPTION = "Ask the user a clarifying question during processing."
CATEGORY: Literal["other"] = "other"
HINTS = ToolHints(open_world=True)


def create_ask_user_tool(
    *,
    name: str = NAME,
    description: str = DESCRIPTION,
    requires_confirmation: bool = False,
) -> AskUserTool:
    """Create a configured AskUserTool instance.

    Args:
        name: Tool name override.
        description: Tool description override.
        requires_confirmation: Whether tool execution needs confirmation.

    Returns:
        Configured AskUserTool instance.
    """
    return AskUserTool(
        name=name,
        description=description,
        category=CATEGORY,
        hints=HINTS,
        requires_confirmation=requires_confirmation,
    )
