"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between pydantic-ai message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never, overload
from urllib.parse import unquote, urlparse

from pydantic import HttpUrl
from pydantic_ai import AudioUrl, BinaryContent, BinaryImage, DocumentUrl, ImageUrl, VideoUrl

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
from agentpool.log import get_logger
from agentpool_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec.asyn import AsyncFileSystem
    from pydantic_ai import UserContent

    from acp.schema import ContentBlock, McpServer
    from agentpool.messaging import MessageNode
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode

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
    """Convert ACP McpServer to native MCPServerConfig.

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


async def _get_resource_context(
    uri: str,
    mime_type: str | None,
    fs: AsyncFileSystem | None,
) -> str | None:
    """Get context for a resource (file or directory).

    Args:
        uri: Resource URI (file:// or zed://)
        mime_type: MIME type hint from client
        fs: Filesystem to use for accessing files

    Returns:
        Context string if applicable, None otherwise
    """
    # Only process file:// URIs for now
    if not uri.startswith("file://"):
        return None

    # Parse the file path from URI
    parsed = urlparse(uri)
    path_str = unquote(parsed.path)
    path = Path(path_str)

    # Use the context_generation module
    from agentpool.repomap import get_resource_context

    # Limit to 50 files for ACP to avoid excessive network calls
    return await get_resource_context(path, fs=fs, max_files_to_read=50)


async def from_acp_content(
    blocks: Sequence[ContentBlock],
    fs: AsyncFileSystem | None = None,
) -> Sequence[UserContent]:
    """Convert ACP content blocks to pydantic-ai UserContent objects.

    Args:
        blocks: List of ACP ContentBlock objects
        fs: Optional filesystem for accessing file/directory content

    Returns:
        List of pydantic-ai UserContent objects (str, ImageUrl, BinaryContent, etc.)
    """
    content: list[UserContent] = []
    logger.info("Processing content blocks", block_count=len(blocks))

    for block in blocks:
        logger.info("Processing block", block_type=type(block).__name__)
        match block:
            case TextContentBlock(text=text):
                content.append(text)

            case ImageContentBlock(data=data, mime_type=mime_type):
                # ACP image data is base64 encoded
                binary_data = base64.b64decode(data)
                content.append(BinaryImage(data=binary_data, media_type=mime_type))

            case AudioContentBlock(data=data, mime_type=mime_type):
                binary_data = base64.b64decode(data)
                content.append(BinaryContent(data=binary_data, media_type=mime_type))

            case ResourceContentBlock(uri=uri, mime_type=mime_type):
                # Convert to appropriate URL type based on MIME type
                if mime_type:
                    if mime_type.startswith("image/"):
                        content.append(ImageUrl(url=uri))
                    elif mime_type.startswith("audio/"):
                        content.append(AudioUrl(url=uri))
                    elif mime_type.startswith("video/"):
                        content.append(VideoUrl(url=uri))
                    elif mime_type == "application/pdf":
                        content.append(DocumentUrl(url=uri))
                    else:
                        # Generic resource - convert to text link and try to add context
                        content.append(format_uri_as_link(uri))
                        context = await _get_resource_context(uri, mime_type, fs)
                        if context:
                            content.append(f'\n<context ref="{uri}">\n{context}\n</context>')
                else:
                    # No MIME type - try to add context for file/directory references
                    content.append(format_uri_as_link(uri))
                    context = await _get_resource_context(uri, mime_type, fs)
                    if context:
                        content.append(f'\n<context ref="{uri}">\n{context}\n</context>')

            case EmbeddedResourceContentBlock(resource=resource):
                match resource:
                    case TextResourceContents(uri=uri, text=text):
                        content.append(format_uri_as_link(uri))
                        # Client provided the text content, use it directly
                        content.append(f'\n<context ref="{uri}">\n{text}\n</context>')
                    case BlobResourceContents(blob=blob, mime_type=mime_type):
                        # Convert embedded binary to appropriate content type
                        binary_data = base64.b64decode(blob)
                        if mime_type and mime_type.startswith("image/"):
                            content.append(BinaryImage(data=binary_data, media_type=mime_type))
                        elif mime_type and mime_type.startswith("audio/"):
                            content.append(BinaryContent(data=binary_data, media_type=mime_type))
                        elif mime_type == "application/pdf":
                            content.append(
                                BinaryContent(data=binary_data, media_type="application/pdf")
                            )
                        else:
                            # Unknown binary type - describe it
                            formatted_uri = format_uri_as_link(resource.uri)
                            content.append(f"Binary Resource: {formatted_uri}")

    return content


def agent_to_mode(agent: MessageNode[Any, Any]) -> SessionMode:
    """Convert agent to a session mode (deprecated - use get_confirmation_modes)."""
    desc = agent.description or f"Switch to {agent.name} agent"
    return SessionMode(id=agent.name, name=agent.display_name, description=desc)


def get_confirmation_modes() -> list[SessionMode]:
    """Get available tool confirmation modes as ACP session modes.

    Returns standard ACP-compatible modes for tool confirmation levels.
    """
    return [
        SessionMode(
            id="default",
            name="Default",
            description="Require confirmation for tools marked as needing it",
        ),
        SessionMode(
            id="acceptEdits",
            name="Accept Edits",
            description="Auto-approve all tool calls without confirmation",
        ),
    ]


def mode_id_to_confirmation_mode(mode_id: str) -> ToolConfirmationMode | None:
    """Map ACP mode ID to ToolConfirmationMode.

    Returns:
        ToolConfirmationMode value or None if mode_id is invalid
    """
    mapping: dict[str, ToolConfirmationMode] = {
        "default": "per_tool",
        "acceptEdits": "never",
        "bypassPermissions": "never",
        # "plan": "..."
    }
    return mapping.get(mode_id)


def confirmation_mode_to_mode_id(mode: ToolConfirmationMode) -> str:
    """Map ToolConfirmationMode to ACP mode ID.

    Args:
        mode: Tool confirmation mode

    Returns:
        ACP mode ID string
    """
    mapping: dict[ToolConfirmationMode, str] = {
        "per_tool": "default",
        "always": "default",  # No direct ACP equivalent, use default (requires confirmation)
        "never": "acceptEdits",
    }
    return mapping.get(mode, "default")
