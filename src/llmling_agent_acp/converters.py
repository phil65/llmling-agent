"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between llmling-agent message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, overload

from pydantic import HttpUrl
from pydantic_ai import (
    BinaryContent,
    FileUrl,
    ToolReturn,
)

from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    SessionMode,
    SseMcpServer,
    StdioMcpServer,
    TextContentBlock,
    TextResourceContents,
)
from llmling_agent.log import get_logger
from llmling_agent_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import UserContent

    from acp.schema import ContentBlock, McpServer
    from llmling_agent import Agent
    from llmling_agent.models.content import BaseContent
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


DEFAULT_PERMISSION_OPTIONS = [
    PermissionOption(option_id="allow_once", name="Allow Once", kind="allow_once"),
    PermissionOption(option_id="deny_once", name="Deny Once", kind="reject_once"),
    PermissionOption(option_id="allow_always", name="Always Allow", kind="allow_always"),
    PermissionOption(option_id="deny_always", name="Always Deny", kind="reject_always"),
]


@overload
def convert_acp_mcp_server_to_config(
    acp_server: HttpMcpServer,
) -> StreamableHTTPMCPServerConfig: ...


@overload
def convert_acp_mcp_server_to_config(
    acp_server: SseMcpServer,
) -> SSEMCPServerConfig: ...


@overload
def convert_acp_mcp_server_to_config(
    acp_server: StdioMcpServer,
) -> StdioMCPServerConfig: ...


@overload
def convert_acp_mcp_server_to_config(acp_server: McpServer) -> MCPServerConfig: ...


def convert_acp_mcp_server_to_config(acp_server: McpServer) -> MCPServerConfig:
    """Convert ACP McpServer to llmling MCPServerConfig.

    Args:
        acp_server: ACP McpServer object from session/new request

    Returns:
        MCPServerConfig instance
    """
    match acp_server:
        case StdioMcpServer(name=name, command=cmd, args=args, env=env_vars):
            env = {var.name: var.value for var in env_vars}
            return StdioMCPServerConfig(name=name, command=cmd, args=list(args), env=env)
        case SseMcpServer(name=name, url=url, headers=headers):
            h = {h.name: h.value for h in headers}
            return SSEMCPServerConfig(name=name, url=HttpUrl(url), headers=h)
        case HttpMcpServer(name=name, url=url, headers=headers):
            h = {h.name: h.value for h in acp_server.headers}
            return StreamableHTTPMCPServerConfig(name=name, url=HttpUrl(url), headers=h)
        case _:
            msg = f"Unsupported MCP server type: {type(acp_server)}"
            raise ValueError(msg)


def format_uri_as_link(uri: str) -> str:
    """Format URI as markdown-style link similar to other ACP implementations.

    Args:
        uri: URI to format (file://, zed://, etc.)

    Returns:
        Markdown-style link in format [@name](uri)
    """
    if uri.startswith("file://"):
        path = uri[7:]  # Remove "file://"
        name = path.split("/")[-1] or path
        return f"[@{name}]({uri})"
    if uri.startswith("zed://"):
        parts = uri.split("/")
        name = parts[-1] or uri
        return f"[@{name}]({uri})"
    return uri


def from_content_blocks(blocks: Sequence[ContentBlock]) -> Sequence[str | BaseContent]:
    """Convert ACP content blocks to structured content objects.

    Args:
        blocks: List of ACP ContentBlock objects

    Returns:
        List of content objects (str for text, Content objects for rich media)
    """
    from llmling_agent.models.content import AudioBase64Content, ImageBase64Content

    content: list[str | BaseContent] = []

    for block in blocks:
        match block:
            case TextContentBlock(text=text):
                content.append(text)
            case ImageContentBlock(data=data, mime_type=mime_type):
                content.append(ImageBase64Content(data=data, mime_type=mime_type))
            case AudioContentBlock(data=data, mime_type=mime_type):
                # Audio always has data
                format_type = mime_type.split("/")[-1] if mime_type else "mp3"
                content.append(AudioBase64Content(data=data, format=format_type))

            case ResourceContentBlock(name=name, description=description, uri=uri):
                # Resource links - convert to text for now
                parts = [f"Resource: {name}"]
                if description:
                    parts.append(f"Description: {description}")
                parts.append(f"URI: {format_uri_as_link(uri)}")
                content.append("\n".join(parts))
            case EmbeddedResourceContentBlock(resource=resource):
                match resource:
                    case TextResourceContents(uri=uri, text=text):
                        content.append(format_uri_as_link(uri))
                        content.append(f'\n<context ref="{uri}">\n{text}\n</context>')
                    case _:
                        # Binary resource - just describe it with formatted URI
                        formatted_uri = format_uri_as_link(resource.uri)
                        content.append(f"Binary Resource: {formatted_uri}")

    return content


def to_acp_content_blocks(  # noqa: PLR0911
    tool_output: (
        ToolReturn | list[ToolReturn] | UserContent | Sequence[UserContent] | None
    ),
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
        case BinaryContent(data=data, media_type=media_type) if media_type.startswith(
            "image/"
        ):
            # Image content - convert binary data to base64
            image_data = base64.b64encode(data).decode("utf-8")
            return [ImageContentBlock(data=image_data, mime_type=media_type)]

        case BinaryContent(data=data, media_type=media_type) if media_type.startswith(
            "audio/"
        ):
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
            return [
                EmbeddedResourceContentBlock(
                    resource=blob_resource,
                    annotations=None,
                )
            ]

        case FileUrl(url=url, kind=kind, media_type=media_type):
            # Handle all URL types with unified logic using FileUrl base class
            from urllib.parse import urlparse

            parsed = urlparse(str(url))

            # Extract resource type from kind (e.g., "image-url" -> "image")
            resource_type = kind.replace("-url", "")

            # Generate name from URL path or use type as fallback
            name = parsed.path.split("/")[-1] if parsed.path else resource_type
            if not name or name == "/":
                name = (
                    f"{resource_type}_{parsed.netloc}" if parsed.netloc else resource_type
                )

            return [
                ResourceContentBlock(
                    uri=str(url),
                    name=name,
                    description=f"{resource_type.title()} resource",
                    mime_type=media_type,  # Uses FileUrl's computed media_type property
                )
            ]

        case _:
            # Everything else - convert to string
            return [TextContentBlock(text=str(tool_output))]


def agent_to_mode(agent: Agent) -> SessionMode:
    return SessionMode(
        id=agent.name,
        name=agent.name,
        description=(agent.description or f"Switch to {agent.name} agent"),
    )
