"""Helper functions for MCP server client operations.

This module contains stateless utility functions that support MCP tool conversion
and content handling for PydanticAI integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import BuiltinToolCallPart, ModelRequest, ToolCallPart

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from mcp.types import ContentBlock, Tool as MCPTool
    from pydantic_ai import ModelMessage


logger = get_logger(__name__)


def mcp_tool_to_fn_schema(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function schema format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
    }


def extract_text_content(mcp_content: list[ContentBlock]) -> str:
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


def extract_tool_call_args(
    messages: list[ModelMessage], tool_call_id: str
) -> dict[str, Any]:
    """Extract tool call arguments from message history.

    Args:
        messages: List of messages to search through
        tool_call_id: ID of the tool call to find

    Returns:
        Dictionary of tool call arguments
    """
    for message in messages:
        if isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if (
                isinstance(part, BuiltinToolCallPart | ToolCallPart)
                and part.tool_call_id == tool_call_id
            ):
                return part.args_as_dict()

    return {}
