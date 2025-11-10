"""Helper functions for MCP server client operations.

This module contains stateless utility functions that support MCP tool conversion
and content handling for PydanticAI integration.
"""

from __future__ import annotations

import base64
import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    BinaryContent,
    BuiltinToolCallPart,
    ModelRequest,
    RunContext,
    ToolCallPart,
)

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastmcp import Client
    from mcp.types import (
        BlobResourceContents,
        ContentBlock,
        TextResourceContents,
        Tool as MCPTool,
    )
    from pydantic_ai import ModelMessage


logger = get_logger(__name__)


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


async def convert_mcp_content(
    mcp_content: Sequence[ContentBlock | TextResourceContents | BlobResourceContents],
    client: Client | None = None,
) -> list[str | BinaryContent]:
    """Convert MCP content blocks to PydanticAI content types.

    If a FastMCP client is given, this function will try to resolve the ResourceLinks.

    """
    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )
    from pydantic_ai import BinaryImage, DocumentUrl

    contents: list[Any] = []

    for block in mcp_content:
        match block:
            case TextContent(text=text):
                contents.append(text)
            case TextResourceContents(text=text):
                contents.append(text)
            case ImageContent(data=data, mimeType=mime_type):
                decoded_data = base64.b64decode(data)
                img = BinaryImage(data=decoded_data, media_type=mime_type)
                contents.append(img)
            case AudioContent(data=data, mimeType=mime_type):
                decoded_data = base64.b64decode(data)
                content = BinaryContent(data=decoded_data, media_type=mime_type)
                contents.append(content)
            case BlobResourceContents(blob=blob):
                decoded_data = base64.b64decode(blob)
                mime = "application/octet-stream"
                content = BinaryContent(data=decoded_data, media_type=mime)
                contents.append(content)
            case ResourceLink(uri=uri):
                if client:
                    try:
                        res = await client.read_resource(uri)
                        nested = await convert_mcp_content(res)
                        contents.extend(nested)
                    except Exception:  # noqa: BLE001
                        # Fallback to DocumentUrl if reading fails
                        logger.warning("Failed to read resource", uri=uri)
                contents.append(DocumentUrl(url=str(uri)))
            case EmbeddedResource(resource=TextResourceContents(text=text)):
                contents.append(text)
            case EmbeddedResource(resource=BlobResourceContents() as blob_resource):
                contents.append(f"[Binary data: {blob_resource.mimeType}]")
            case _:
                contents.append(str(block))  # Convert anything else to string
    return contents


def content_block_as_text(content: ContentBlock) -> str:  # noqa: PLR0911

    # Convert MCP messages to pydantic-ai parts
    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    match content:
        case TextContent(text=text):
            return text
        case EmbeddedResource(resource=TextResourceContents() as content):
            return content.text
        case EmbeddedResource(resource=BlobResourceContents() as content):
            return f"[Resource: {content.uri}]"
        case EmbeddedResource():
            return f"[Resource: {content.uri}]"
        case ResourceLink(uri=uri, description=desc):
            return (
                f"[Resource Link: {uri}] - {desc}" if desc else f"[Resource Link: {uri}]"
            )
        case ImageContent(mimeType=mime_type):
            return f"[Image: {mime_type}]"
        case AudioContent(mimeType=mime_type):
            return f"[Audio: {mime_type}]"
    msg = "Unexpected content type"
    raise TypeError(msg, type=type(content).__name__)
