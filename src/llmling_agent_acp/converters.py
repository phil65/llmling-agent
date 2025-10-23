"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between llmling-agent message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from acp.acp_types import HttpMcpServer, SseMcpServer, StdioMcpServer
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    ResourceLink,
    SessionNotification,
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

    from acp.acp_types import ContentBlock, MCPServer, ToolCallKind
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
def convert_acp_mcp_server_to_config(acp_server: MCPServer) -> MCPServerConfig: ...


def convert_acp_mcp_server_to_config(acp_server: MCPServer) -> MCPServerConfig:
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
                parts.append(f"URI: {uri}")
                content.append("\n".join(parts))

            case ResourceLink(uri=uri):
                # Format as markdown-style link
                formatted_uri = format_uri_as_link(uri)
                content.append(formatted_uri)

            case EmbeddedResourceContentBlock(resource=resource):
                match resource:
                    case TextResourceContents(uri=uri, text=text):
                        formatted_uri = format_uri_as_link(uri)
                        content.append(formatted_uri)
                        context_block = f'\n<context ref="{uri}">\n{text}\n</context>'
                        content.append(context_block)
                    case _:
                        # Binary resource - just describe it with formatted URI
                        formatted_uri = format_uri_as_link(resource.uri)
                        content.append(f"Binary Resource: {formatted_uri}")

    return content


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
    if any(
        i in name_lower for i in ["write", "save", "edit", "modify", "update"]
    ) and any(i in name_lower for i in ["file", "path", "content"]):
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
