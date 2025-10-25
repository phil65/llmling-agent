"""Helper functions for MCP server client operations.

This module contains stateless utility functions that support MCP tool conversion
and content handling for PydanticAI integration.
"""

from __future__ import annotations

from typing import Any


def convert_mcp_content_to_pydantic(mcp_content: list[Any]) -> list[str | Any]:
    """Convert MCP content blocks to PydanticAI content types.

    Args:
        mcp_content: List of MCP content blocks to convert

    Returns:
        List of PydanticAI-compatible content objects
    """
    import base64

    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )
    from pydantic_ai.messages import BinaryContent

    pydantic_content = []

    for block in mcp_content:
        match block:
            case TextContent(text=text):
                pydantic_content.append(text)
            case TextResourceContents(text=text):
                pydantic_content.append(text)
            case ImageContent(data=data, mimeType=mime_type):
                # MCP data is base64 encoded string, decode it for PydanticAI
                decoded_data = base64.b64decode(data)
                pydantic_content.append(
                    BinaryContent(data=decoded_data, media_type=mime_type)
                )
            case AudioContent(data=data, mimeType=mime_type):
                # MCP data is base64 encoded string, decode it for PydanticAI
                decoded_data = base64.b64decode(data)
                pydantic_content.append(
                    BinaryContent(data=decoded_data, media_type=mime_type)
                )
            case BlobResourceContents(blob=blob):
                # MCP blob is base64 encoded string
                decoded_data = base64.b64decode(blob)
                pydantic_content.append(
                    BinaryContent(
                        data=decoded_data, media_type="application/octet-stream"
                    )
                )
            case ResourceLink(uri=uri, name=name):
                # ResourceLink should be converted to DocumentUrl for PydanticAI
                from pydantic_ai.messages import DocumentUrl

                pydantic_content.append(DocumentUrl(url=str(uri)))
            case EmbeddedResource(resource=resource):
                # EmbeddedResource contains another resource, process recursively
                if hasattr(resource, "contents"):
                    for content in resource.contents:
                        nested_result = convert_mcp_content_to_pydantic([content])
                        pydantic_content.extend(nested_result)
                else:
                    pydantic_content.append(str(resource))
            case _:
                # Convert anything else to string
                pydantic_content.append(str(block))

    return pydantic_content


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
