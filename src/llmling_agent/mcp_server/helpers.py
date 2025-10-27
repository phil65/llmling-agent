"""Helper functions for MCP server client operations.

This module contains stateless utility functions that support MCP tool conversion
and content handling for PydanticAI integration.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext


if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool


def _create_tool_signature_with_context(
    base_signature: inspect.Signature,
) -> inspect.Signature:
    """Create a function signature that includes RunContext as first parameter.

    This is crucial for PydanticAI integration - it expects tools that need RunContext
    to have it as the first parameter with proper annotation. Without this, PydanticAI
    won't pass the RunContext and we lose access to tool_call_id and other context.

    Args:
        base_signature: Original signature from MCP tool schema (tool parameters only)

    Returns:
        New signature: (ctx: RunContext, ...original_params) -> ReturnType

    Example:
        Original: (message: str) -> str
        Result:   (ctx: RunContext, message: str) -> str
    """
    # Create RunContext parameter
    ctx_param = inspect.Parameter(
        "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext
    )
    # Combine with tool parameters
    tool_params = list(base_signature.parameters.values())
    new_params = [ctx_param, *tool_params]

    return base_signature.replace(parameters=new_params)


def _create_tool_annotations_with_context(
    base_annotations: dict[str, Any],
) -> dict[str, Any]:
    """Create function annotations that include RunContext for first parameter.

    Complements _create_tool_signature_with_context by ensuring the annotations
    match the signature. This is required for proper type checking and PydanticAI's
    introspection of tool functions.

    Args:
        base_annotations: Original annotations from MCP tool schema

    Returns:
        New annotations dict with 'ctx': RunContext added to base annotations

    Example:
        Original: {'message': str, 'return': str}
        Result:   {'ctx': RunContext, 'message': str, 'return': str}
    """
    new_annotations = base_annotations.copy()
    new_annotations["ctx"] = RunContext
    return new_annotations


def mcp_tool_to_fn_schema(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function schema format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
    }


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
