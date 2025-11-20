"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between llmling-agent message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never, overload

from pydantic import HttpUrl

from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
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

    from acp.schema import ContentBlock, McpServer
    from llmling_agent import Agent
    from llmling_agent.models.content import BaseContent
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


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
        case _ as unreachable:
            assert_never(unreachable)


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
    from llmling_agent.models.content import (
        AudioBase64Content,
        AudioURLContent,
        ImageBase64Content,
        ImageURLContent,
        PDFBase64Content,
        PDFURLContent,
        VideoURLContent,
    )

    content: list[str | BaseContent] = []

    for block in blocks:
        match block:
            case TextContentBlock(text=text):
                content.append(text)

            case ImageContentBlock(data=data, mime_type=mime_type, uri=uri):
                # Prefer data over URI as per ACP specification
                content.append(ImageBase64Content(data=data, mime_type=mime_type))

            case AudioContentBlock(data=data, mime_type=mime_type):
                format_type = mime_type.split("/")[-1] if mime_type else "mp3"
                content.append(AudioBase64Content(data=data, format=format_type))

            case ResourceContentBlock(uri=uri, description=description, mime_type=mime_type):
                # Convert to appropriate content type based on MIME type
                if mime_type:
                    if mime_type.startswith("image/"):
                        content.append(ImageURLContent(url=uri, description=description))
                    elif mime_type.startswith("audio/"):
                        content.append(AudioURLContent(url=uri, description=description))
                    elif mime_type.startswith("video/"):
                        content.append(VideoURLContent(url=uri, description=description))
                    elif mime_type == "application/pdf":
                        content.append(PDFURLContent(url=uri, description=description))
                    else:
                        # Generic resource - convert to text link
                        content.append(format_uri_as_link(uri))
                else:
                    # No MIME type - fallback to text link
                    content.append(format_uri_as_link(uri))

            case EmbeddedResourceContentBlock(resource=resource):
                match resource:
                    case TextResourceContents(uri=uri, text=text):
                        content.append(format_uri_as_link(uri))
                        content.append(f'\n<context ref="{uri}">\n{text}\n</context>')
                    case BlobResourceContents(blob=blob, mime_type=mime_type):
                        # Convert embedded binary to appropriate content type
                        if mime_type and mime_type.startswith("image/"):
                            content.append(ImageBase64Content(data=blob, mime_type=mime_type))
                        elif mime_type and mime_type.startswith("audio/"):
                            format_type = mime_type.split("/")[-1]
                            content.append(AudioBase64Content(data=blob, format=format_type))
                        elif mime_type == "application/pdf":
                            content.append(PDFBase64Content(data=blob))
                        else:
                            # Unknown binary type - describe it
                            formatted_uri = format_uri_as_link(resource.uri)
                            content.append(f"Binary Resource: {formatted_uri}")

    return content


def agent_to_mode(agent: Agent) -> SessionMode:
    return SessionMode(
        id=agent.name,
        name=agent.name,
        description=(agent.description or f"Switch to {agent.name} agent"),
    )
