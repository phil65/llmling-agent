"""Utility functions for ACP (Agent Client Protocol)."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from acp.acp_types import ToolCallKind


def infer_tool_kind(tool_name: str) -> ToolCallKind:  # noqa: PLR0911
    """Determine the appropriate tool kind based on name.

    Simple substring matching for tool kind inference.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool kind string for ACP protocol
    """
    name_lower = tool_name.lower()
    if any(i in name_lower for i in ["read", "load", "get"]) and any(
        i in name_lower for i in ["file", "path", "content"]
    ):
        return "read"
    if any(
        i in name_lower for i in ["write", "save", "edit", "modify", "update"]
    ) and any(i in name_lower for i in ["file", "path", "content"]):
        return "edit"
    if any(i in name_lower for i in ["delete", "remove", "rm"]):
        return "delete"
    if any(i in name_lower for i in ["move", "rename", "mv"]):
        return "move"
    if any(i in name_lower for i in ["search", "find", "query", "lookup"]):
        return "search"
    if any(i in name_lower for i in ["execute", "run", "exec", "command", "shell"]):
        return "execute"
    if any(i in name_lower for i in ["think", "plan", "reason", "analyze"]):
        return "think"
    if any(i in name_lower for i in ["fetch", "download", "request"]):
        return "fetch"
    return "other"  # Default to other
