"""ACP to native event converters.

This module provides conversion from ACP session updates to native llmling-agent
streaming events, enabling ACPAgent to yield the same event types as native agents.

This is the reverse of the conversion done in acp_server/session.py handle_event().
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, overload

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    PartDeltaEvent,
    VideoUrl,
)
from pydantic_ai.messages import TextPartDelta, ThinkingPartDelta

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    BlobResourceContents,
    ContentToolCallContent,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    ImageContentBlock,
    ResourceContentBlock,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from llmling_agent.agents.events import (
    DiffContentItem,
    LocationContentItem,
    PlanUpdateEvent,
    TerminalContentItem,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import UserContent

    from acp.schema import ContentBlock, SessionUpdate
    from acp.schema.mcp import HttpMcpServer, McpServer, SseMcpServer, StdioMcpServer
    from acp.schema.tool_call import ToolCallContent, ToolCallLocation
    from llmling_agent.agents.events import RichAgentStreamEvent, ToolCallContentItem
    from llmling_agent_config.mcp_server import (
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )


def convert_acp_locations(
    locations: list[ToolCallLocation] | None,
) -> list[LocationContentItem]:
    """Convert ACP ToolCallLocation list to native LocationContentItem list."""
    return [LocationContentItem(path=loc.path, line=loc.line) for loc in locations or []]


def convert_acp_content(content: list[ToolCallContent] | None) -> list[ToolCallContentItem]:
    """Convert ACP ToolCallContent list to native ToolCallContentItem list."""
    if not content:
        return []

    result: list[ToolCallContentItem] = []
    for item in content:
        match item:
            case TerminalToolCallContent(terminal_id=terminal_id):
                result.append(TerminalContentItem(terminal_id=terminal_id))
            case FileEditToolCallContent(path=path, old_text=old_text, new_text=new_text):
                result.append(DiffContentItem(path=path, old_text=old_text, new_text=new_text))
            case ContentToolCallContent(content=TextContentBlock(text=text)):
                from llmling_agent.agents.events import TextContentItem

                result.append(TextContentItem(text=text))
    return result


def convert_to_acp_content(
    prompts: Sequence[UserContent],
) -> list[ContentBlock]:
    """Convert pydantic-ai UserContent to ACP ContentBlock format.

    Handles text, images, audio, video, and document content types.

    Args:
        prompts: pydantic-ai UserContent items

    Returns:
        List of ACP ContentBlock items
    """
    content_blocks: list[ContentBlock] = []

    for item in prompts:
        match item:
            case str(text):
                content_blocks.append(TextContentBlock(text=text))

            case BinaryImage(data=data, media_type=media_type):
                encoded = base64.b64encode(data).decode("utf-8")
                content_blocks.append(ImageContentBlock(data=encoded, mime_type=media_type))

            case BinaryContent(data=data, media_type=media_type):
                encoded = base64.b64encode(data).decode("utf-8")
                # Handle different media types
                if media_type and media_type.startswith("image/"):
                    content_blocks.append(ImageContentBlock(data=encoded, mime_type=media_type))
                elif media_type and media_type.startswith("audio/"):
                    content_blocks.append(AudioContentBlock(data=encoded, mime_type=media_type))
                elif media_type == "application/pdf":
                    blob_resource = BlobResourceContents(
                        blob=encoded,
                        mime_type="application/pdf",
                        uri=f"data:application/pdf;base64,{encoded[:50]}...",
                    )
                    content_blocks.append(EmbeddedResourceContentBlock(resource=blob_resource))
                else:
                    # Generic binary as embedded resource
                    blob_resource = BlobResourceContents(
                        blob=encoded,
                        mime_type=media_type or "application/octet-stream",
                        uri=f"data:{media_type or 'application/octet-stream'};base64,...",
                    )
                    content_blocks.append(EmbeddedResourceContentBlock(resource=blob_resource))

            case ImageUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=str(url),
                        name="Image",
                        mime_type=media_type or "image/jpeg",
                    )
                )

            case AudioUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=str(url),
                        name="Audio",
                        mime_type=media_type or "audio/wav",
                        description="Audio content",
                    )
                )

            case DocumentUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=str(url),
                        name="Document",
                        mime_type=media_type or "application/pdf",
                        description="Document",
                    )
                )

            case VideoUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=str(url),
                        name="Video",
                        mime_type=media_type or "video/mp4",
                        description="Video content",
                    )
                )

    return content_blocks


def acp_to_native_event(update: SessionUpdate) -> RichAgentStreamEvent[Any] | None:  # noqa: PLR0911
    """Convert ACP session update to native streaming event.

    Args:
        update: ACP SessionUpdate from session/update notification

    Returns:
        Corresponding native event, or None if no mapping exists
    """
    match update:
        # Text message chunks -> PartDeltaEvent with TextPartDelta
        case AgentMessageChunk(content=TextContentBlock(text=text)):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=text))

        # Thought chunks -> PartDeltaEvent with ThinkingPartDelta
        case AgentThoughtChunk(content=TextContentBlock(text=text)):
            return PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=text))

        # User message echo - could emit as PartStartEvent if needed
        case UserMessageChunk():
            return None  # Usually ignored

        # Tool call start -> ToolCallStartEvent
        case ToolCallStart() as tc:
            return ToolCallStartEvent(
                tool_call_id=tc.tool_call_id,
                tool_name=tc.title,  # ACP uses title, not separate tool_name
                title=tc.title,
                kind=tc.kind or "other",
                content=convert_acp_content(list(tc.content) if tc.content else None),
                locations=convert_acp_locations(list(tc.locations) if tc.locations else None),
                raw_input=tc.raw_input or {},
            )

        # Tool call progress -> ToolCallProgressEvent
        case ToolCallProgress() as tc:
            items = convert_acp_content(list(tc.content) if tc.content else None)

            return ToolCallProgressEvent(
                tool_call_id=tc.tool_call_id,
                status=tc.status or "in_progress",
                title=tc.title,
                items=items,
                message=str(tc.raw_output) if tc.raw_output else None,
            )

        # Plan update -> PlanUpdateEvent
        case AgentPlanUpdate(entries=acp_entries):
            from llmling_agent.resource_providers.plan_provider import PlanEntry

            native_entries = [
                PlanEntry(content=e.content, priority=e.priority, status=e.status)
                for e in acp_entries
            ]
            return PlanUpdateEvent(entries=native_entries)

        case _:
            return None


def extract_text_from_update(update: SessionUpdate) -> str | None:
    """Extract plain text content from an ACP session update.

    Args:
        update: ACP SessionUpdate

    Returns:
        Text content if this is a text-bearing update, None otherwise
    """
    match update:
        case AgentMessageChunk(content=TextContentBlock(text=text)):
            return text
        case AgentThoughtChunk(content=TextContentBlock(text=text)):
            return text
        case _:
            return None


def is_text_update(update: SessionUpdate) -> bool:
    """Check if this update contains text content."""
    return extract_text_from_update(update) is not None


@overload
def mcp_config_to_acp(config: StdioMCPServerConfig) -> StdioMcpServer | None: ...


@overload
def mcp_config_to_acp(config: SSEMCPServerConfig) -> SseMcpServer | None: ...


@overload
def mcp_config_to_acp(config: StreamableHTTPMCPServerConfig) -> HttpMcpServer | None: ...


@overload
def mcp_config_to_acp(config: MCPServerConfig) -> McpServer | None: ...


def mcp_config_to_acp(config: MCPServerConfig) -> McpServer | None:
    """Convert llmling MCPServerConfig to ACP McpServer format.

    Args:
        config: llmling-agent MCP server configuration

    Returns:
        ACP-compatible McpServer instance, or None if conversion not possible
    """
    from pydantic import HttpUrl

    from acp.schema.common import EnvVariable
    from acp.schema.mcp import HttpMcpServer, SseMcpServer, StdioMcpServer
    from llmling_agent_config.mcp_server import (
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    match config:
        case StdioMCPServerConfig(command=command, args=args):
            env_vars = config.get_env_vars() if hasattr(config, "get_env_vars") else {}
            return StdioMcpServer(
                name=config.name or command,
                command=command,
                args=list(args) if args else [],
                env=[EnvVariable(name=k, value=v) for k, v in env_vars.items()],
            )
        case SSEMCPServerConfig(url=url):
            return SseMcpServer(
                name=config.name or str(url),
                url=HttpUrl(str(url)),
                headers=[],
            )
        case StreamableHTTPMCPServerConfig(url=url):
            return HttpMcpServer(
                name=config.name or str(url),
                url=HttpUrl(str(url)),
                headers=[],
            )
        case _:
            return None


def mcp_configs_to_acp(configs: Sequence[MCPServerConfig]) -> list[McpServer]:
    """Convert a sequence of MCPServerConfig to ACP McpServer list.

    Args:
        configs: Sequence of llmling-agent MCP server configurations

    Returns:
        List of ACP-compatible McpServer instances (skips unconvertible configs)
    """
    return [converted for config in configs if (converted := mcp_config_to_acp(config)) is not None]
