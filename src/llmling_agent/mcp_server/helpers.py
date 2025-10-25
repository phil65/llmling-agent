"""Helper functions for MCP server client operations.

This module contains stateless utility functions that support MCP tool conversion
and content handling for PydanticAI integration.
"""

from __future__ import annotations

from typing import Any


def extract_text_content(mcp_content: list[Any]) -> str:
    """Extract text content from MCP content blocks.

    Args:
        mcp_content: List of MCP content blocks

    Returns:
        First available text content or fallback string
    """
    from mcp.types import TextContent

    for block in mcp_content:
        match block:
            case TextContent(text=text):
                return text

    # Fallback: stringify the content
    return str(mcp_content[0]) if mcp_content else "Tool executed successfully"


def extract_tool_call_args(messages: list[Any], tool_call_id: str) -> dict[str, Any]:
    """Extract tool call arguments from message history.

    Args:
        messages: List of messages to search through
        tool_call_id: ID of the tool call to find

    Returns:
        Dictionary of tool call arguments
    """
    for message in messages:
        if hasattr(message, "parts"):
            for part in message.parts:
                if (
                    hasattr(part, "tool_call_id")
                    and part.tool_call_id == tool_call_id
                    and hasattr(part, "args_as_dict")
                ):
                    return part.args_as_dict()
        elif hasattr(message, "content"):
            # Handle different message structures
            for content in message.content:
                if (
                    hasattr(content, "tool_call_id")
                    and content.tool_call_id == tool_call_id
                    and hasattr(content, "args")
                ):
                    return content.args
    return {}
