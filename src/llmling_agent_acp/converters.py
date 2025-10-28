"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between llmling-agent message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, overload

from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    ContentToolCallContent,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    SessionNotification,
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

    from acp.schema import ContentBlock, McpServer
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
            return SSEMCPServerConfig(name=name, url=url, headers=h)
        case HttpMcpServer(name=name, url=url, headers=headers):
            h = {h.name: h.value for h in acp_server.headers}
            return StreamableHTTPMCPServerConfig(name=name, url=url, headers=h)
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
    tool_output: Any,
) -> list[TextContentBlock | ImageContentBlock | AudioContentBlock]:
    """Convert pydantic-ai tool output to raw ACP content blocks.

    Returns unwrapped content blocks that can be used directly or wrapped
    in ContentToolCallContent as needed.

    Args:
        tool_output: Output from pydantic-ai tool execution

    Returns:
        List of content blocks (TextContentBlock, ImageContentBlock, AudioContentBlock)
    """
    try:
        # Import pydantic-ai types only when needed to avoid dependency issues
        from pydantic_ai import (
            AudioUrl,
            BinaryContent,
            DocumentUrl,
            ImageUrl,
            ToolReturn,
            VideoUrl,
        )
    except ImportError:
        # Fallback if pydantic-ai not available - convert to text
        if tool_output is not None:
            return [TextContentBlock(text=str(tool_output))]
        return []

    if tool_output is None:
        return []

    # Handle ToolReturn objects with separate content field
    if isinstance(tool_output, ToolReturn):
        blocks: list[TextContentBlock | ImageContentBlock | AudioContentBlock] = []

        # Add the return value as text
        if tool_output.return_value is not None:
            blocks.append(TextContentBlock(text=str(tool_output.return_value)))

        # Add any multimodal content
        if tool_output.content:
            content_list = (
                tool_output.content
                if isinstance(tool_output.content, list)
                else [tool_output.content]
            )
            for content_item in content_list:
                blocks.extend(to_acp_content_blocks(content_item))

        return blocks

    # Handle lists of content
    if isinstance(tool_output, list):
        blocks = []
        for item in tool_output:
            blocks.extend(to_acp_content_blocks(item))
        return blocks

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
            # Other binary content - describe as text for now
            size_mb = len(data) / (1024 * 1024)
            return [
                TextContentBlock(text=f"Binary content ({media_type}): {size_mb:.2f} MB")
            ]

        case ImageUrl(url=url):
            # Image URL - create markdown image link
            return [TextContentBlock(text=f"![Image]({url})")]

        case AudioUrl(url=url):
            # Audio URL - create markdown link
            return [TextContentBlock(text=f"[Audio]({url})")]

        case DocumentUrl(url=url):
            # Document URL - create markdown link
            return [TextContentBlock(text=f"[Document]({url})")]

        case VideoUrl(url=url):
            # Video URL - create markdown link
            return [TextContentBlock(text=f"[Video]({url})")]

        case _:
            # Everything else - convert to string
            return [TextContentBlock(text=str(tool_output))]


def to_tool_call_contents(tool_output: Any) -> list[ContentToolCallContent]:
    """Convert pydantic-ai tool output to ACP content blocks.

    Handles multimodal content from pydantic-ai including images, audio, binary content,
    URLs, and ToolReturn objects with separate content fields.

    Args:
        tool_output: Output from pydantic-ai tool execution

    Returns:
        List of ContentToolCallContent blocks for ACP
    """
    raw_blocks = to_acp_content_blocks(tool_output)
    return [ContentToolCallContent(content=block) for block in raw_blocks]


def to_agent_text_notification(
    response: str, session_id: str
) -> SessionNotification | None:
    """Convert agent response text to ACP session notification.

    Args:
        response: Response text from llmling agent
        session_id: ACP session identifier

    Returns:
        SessionNotification with agent text message, or None if response is empty
    """
    if not response.strip():
        return None

    update = AgentMessageChunk(content=TextContentBlock(text=response))
    return SessionNotification(session_id=session_id, update=update)
