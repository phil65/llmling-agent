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


async def _get_resource_context(uri: str, mime_type: str | None) -> str | None:
    """Get context for a resource (file or directory).

    Args:
        uri: Resource URI (file:// or zed://)
        mime_type: MIME type hint from client

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

    # Check if path exists
    if not path.exists():
        logger.debug("Path does not exist", path=str(path))
        return None

    try:
        if path.is_dir():
            # Generate repomap for directories
            return await _generate_directory_repomap(path)
        # Generate file outline/content
        return await _generate_file_context(path)
    except Exception as e:
        logger.warning("Failed to generate context", uri=uri, error=str(e))
        return None


async def _generate_directory_repomap(path: Path) -> str | None:
    """Generate a repomap for a directory.

    Args:
        path: Directory path

    Returns:
        Repomap string or None if generation fails
    """
    logger.info("Generating directory repomap", path=str(path))

    try:
        from fsspec.implementations.local import LocalFileSystem

        from agentpool.repomap import RepoMap

        fs = LocalFileSystem()
        repo_map = RepoMap(fs, root_path=str(path), max_tokens=4096)

        # Find all source files in the directory (non-recursive for now)
        files: list[str] = []
        for item in path.iterdir():
            if item.is_file() and item.suffix in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                files.append(str(item))

        if not files:
            # No source files found
            logger.debug("No source files found in directory", path=str(path))
            return f"Directory: {path}\n(No source files found)"

        # Generate the map
        logger.info("Generating repomap", file_count=len(files), path=str(path))
        map_content = await repo_map.get_map(files)

        if map_content:
            logger.info("Successfully generated repomap", content_length=len(map_content), path=str(path))
            logger.info(
                "Successfully generated repomap", content_length=len(map_content), path=str(path)
            )
            header = f"# Repository map for {path.name}\n\n"
            return header + map_content

        logger.warning("Repomap generation returned empty", path=str(path))
        return f"Directory: {path}"

    except Exception as e:
        logger.warning("Failed to generate directory repomap", path=str(path), error=str(e))
        return None


async def _generate_file_context(path: Path) -> str | None:
    """Generate context for a file (outline or content based on size).

    Args:
        path: File path

    Returns:
        File context string or None if generation fails
    """
    logger.info("Generating file context", path=str(path))

    try:
        from agentpool.repomap import get_file_map_from_content

        # Read file content
        content = path.read_text(encoding="utf-8", errors="ignore")

        # For large files (>8KB ~ 2000 tokens), try to generate outline
        LARGE_FILE_THRESHOLD = 8192
        if len(content) > LARGE_FILE_THRESHOLD:
            logger.info("File is large, generating outline", path=str(path), size=len(content))
            # Try to get structure map
            file_map = get_file_map_from_content(str(path), content)
            if file_map:
                logger.info("Generated file outline", path=str(path), outline_length=len(file_map))
                return f"# File outline for {path}\n\n{file_map}"
            else:
                # No outline available, truncate content
                logger.debug("No outline available, truncating", path=str(path))
                from agentpool.repomap import truncate_with_notice

        # Small file, return full content
        logger.debug("File is small, returning full content", path=str(path), size=len(content))
        return content

    except Exception as e:
        logger.warning("Failed to generate file context", path=str(path), error=str(e))
        return None


async def _enhance_directory_context(uri: str, text: str) -> str:
    """Enhance directory context with repomap if text is minimal.

    Args:
        uri: Resource URI (file:// or zed://)
        text: Original context text from IDE

    Returns:
        Enhanced text with repomap if applicable, otherwise original text
    """
    logger.info("Enhancing directory context", uri=uri, text_length=len(text))

    # Only process file:// URIs for now
    if not uri.startswith("file://"):
        logger.debug("Skipping non-file URI", uri=uri)
        return text

    # Parse the file path from URI
    parsed = urlparse(uri)
    path_str = unquote(parsed.path)
    path = Path(path_str)

    # Check if it's a directory
    if not path.exists():
        logger.debug("Path does not exist", path=str(path))
        return text

    if not path.is_dir():
        logger.debug("Path is not a directory", path=str(path))
        return text

    logger.info("Detected directory reference", path=str(path))

    # If text already has substantial content (more than just outline headers),
    # don't override it
    if text and len(text.strip()) > 200:
        return text

    # Generate repomap for the directory
    try:
        from fsspec.implementations.local import LocalFileSystem

        from agentpool.repomap import RepoMap

        fs = LocalFileSystem()
        repo_map = RepoMap(fs, root_path=str(path), max_tokens=4096)

        # Find all source files in the directory (non-recursive for now)
        files: list[str] = []
        for item in path.iterdir():
            if item.is_file() and item.suffix in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                files.append(str(item))

        if not files:
            # No source files found, return original text
            return text or f"Directory: {path}\n(No source files found)"

        # Generate the map
        logger.info("Generating repomap", file_count=len(files), path=str(path))
        map_content = await repo_map.get_map(files)

        if map_content:
            logger.info(
                "Successfully generated repomap", content_length=len(map_content), path=str(path)
            )
            header = f"# Repository map for {path.name}\n\n"
            return header + map_content

        logger.warning("Repomap generation returned empty", path=str(path))
        return text or f"Directory: {path}"

    except Exception as e:
        # If repomap generation fails, fall back to original text
        logger.warning("Failed to generate repomap", uri=uri, error=str(e))
        return text


async def from_acp_content(blocks: Sequence[ContentBlock]) -> Sequence[UserContent]:
    """Convert ACP content blocks to pydantic-ai UserContent objects.

    Args:
        blocks: List of ACP ContentBlock objects

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
                        context = await _get_resource_context(uri, mime_type)
                        if context:
                            content.append(f'\n<context ref="{uri}">\n{context}\n</context>')
                else:
                    # No MIME type - try to add context for file/directory references
                    content.append(format_uri_as_link(uri))
                    context = await _get_resource_context(uri, mime_type)
                    if context:
                        content.append(f'\n<context ref="{uri}">\n{context}\n</context>')

            case EmbeddedResourceContentBlock(resource=resource):
                match resource:
                    case TextResourceContents(uri=uri, text=text):
                        content.append(format_uri_as_link(uri))
                        # Enhance directory references with repomap if text is minimal
                        enhanced_text = await _enhance_directory_context(uri, text)
                        content.append(f'\n<context ref="{uri}">\n{enhanced_text}\n</context>')
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
