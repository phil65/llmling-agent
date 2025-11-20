"""Utility functions for ACP (Agent Client Protocol)."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pydantic_ai import BinaryContent, FileUrl, ToolReturn

from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    TextContentBlock,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import UserContent

    from acp.schema import ContentBlock, ToolCallKind


logger = get_logger(__name__)


DEFAULT_PERMISSION_OPTIONS = [
    PermissionOption(option_id="allow_once", name="Allow Once", kind="allow_once"),
    PermissionOption(option_id="deny_once", name="Deny Once", kind="reject_once"),
    PermissionOption(option_id="allow_always", name="Always Allow", kind="allow_always"),
    PermissionOption(option_id="deny_always", name="Always Deny", kind="reject_always"),
]


def to_acp_content_blocks(  # noqa: PLR0911
    tool_output: (ToolReturn | list[ToolReturn] | UserContent | Sequence[UserContent] | None),
) -> list[ContentBlock]:
    """Convert pydantic-ai tool output to raw ACP content blocks.

    Returns unwrapped content blocks that can be used directly or wrapped
    in ContentToolCallContent as needed.

    Args:
        tool_output: Output from pydantic-ai tool execution

    Returns:
        List of ContentBlock objects
    """
    if tool_output is None:
        return []

    # Handle ToolReturn objects with separate content field
    if isinstance(tool_output, ToolReturn):
        result_blocks: list[ContentBlock] = []

        # Add the return value as text
        if tool_output.return_value is not None:
            result_blocks.append(TextContentBlock(text=str(tool_output.return_value)))

        # Add any multimodal content
        if tool_output.content:
            content_list = (
                tool_output.content
                if isinstance(tool_output.content, list)
                else [tool_output.content]
            )
            for content_item in content_list:
                result_blocks.extend(to_acp_content_blocks(content_item))

        return result_blocks

    # Handle lists of content
    if isinstance(tool_output, list):
        list_blocks: list[ContentBlock] = []
        for item in tool_output:
            list_blocks.extend(to_acp_content_blocks(item))
        return list_blocks

    # Handle multimodal content types
    match tool_output:
        case BinaryContent(data=data, media_type=media_type) if media_type.startswith("image/"):
            # Image content - convert binary data to base64
            image_data = base64.b64encode(data).decode("utf-8")
            return [ImageContentBlock(data=image_data, mime_type=media_type)]

        case BinaryContent(data=data, media_type=media_type) if media_type.startswith("audio/"):
            # Audio content - convert binary data to base64
            audio_data = base64.b64encode(data).decode("utf-8")
            return [AudioContentBlock(data=audio_data, mime_type=media_type)]

        case BinaryContent(data=data, media_type=media_type):
            # Other binary content - embed as blob resource
            blob_data = base64.b64encode(data).decode("utf-8")
            blob_resource = BlobResourceContents(
                blob=blob_data,
                mime_type=media_type,
                uri=f"data:{media_type};base64,{blob_data[:50]}...",
            )
            return [EmbeddedResourceContentBlock(resource=blob_resource)]

        case FileUrl(url=url, kind=kind, media_type=media_type):
            # Handle all URL types with unified logic using FileUrl base class
            from urllib.parse import urlparse

            parsed = urlparse(str(url))

            # Extract resource type from kind (e.g., "image-url" -> "image")
            resource_type = kind.replace("-url", "")

            # Generate name from URL path or use type as fallback
            name = parsed.path.split("/")[-1] if parsed.path else resource_type
            if not name or name == "/":
                name = f"{resource_type}_{parsed.netloc}" if parsed.netloc else resource_type

            return [
                ResourceContentBlock(
                    uri=str(url),
                    name=name,
                    description=f"{resource_type.title()} resource",
                    mime_type=media_type,
                )
            ]

        case _:
            # Everything else - convert to string
            return [TextContentBlock(text=str(tool_output))]


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
    if any(i in name_lower for i in ["write", "save", "edit", "modify", "update"]) and any(
        i in name_lower for i in ["file", "path", "content"]
    ):
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
