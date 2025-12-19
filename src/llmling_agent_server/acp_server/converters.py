"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between pydantic-ai message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, assert_never, overload

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
    from llmling_agent.messaging import MessageNode
    from llmling_agent_config.mcp_server import MCPServerConfig
    from llmling_agent_config.nodes import ToolConfirmationMode

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


def from_acp_content(blocks: Sequence[ContentBlock]) -> Sequence[UserContent]:
    """Convert ACP content blocks to pydantic-ai UserContent objects.

    Args:
        blocks: List of ACP ContentBlock objects

    Returns:
        List of pydantic-ai UserContent objects (str, ImageUrl, BinaryContent, etc.)
    """
    content: list[UserContent] = []

    for block in blocks:
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
