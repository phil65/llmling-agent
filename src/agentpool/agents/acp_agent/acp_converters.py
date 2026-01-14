"""ACP to native event converters.

This module provides conversion from ACP session updates to native agentpool
streaming events, enabling ACPAgent to yield the same event types as native agents.

This is the reverse of the conversion done in acp_server/session.py handle_event().
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, assert_never, overload

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
    VideoUrl,
)

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
from agentpool.agents.events import (
    DiffContentItem,
    LocationContentItem,
    PlanUpdateEvent,
    TerminalContentItem,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import FinishReason, UserContent

    from acp.schema import (
        ContentBlock,
        HttpMcpServer,
        McpServer,
        SessionConfigOption,
        SessionModelState,
        SessionModeState,
        SessionUpdate,
        SseMcpServer,
        StdioMcpServer,
        StopReason,
        ToolCallContent,
        ToolCallLocation,
    )
    from agentpool.agents.events import RichAgentStreamEvent, ToolCallContentItem
    from agentpool.agents.modes import ModeCategory, ModeInfo
    from agentpool_config.mcp_server import (
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

STOP_REASON_MAP: dict[StopReason, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "content_filter",
    "cancelled": "error",
}


def get_modes(
    config_options: list[SessionConfigOption],
    available_modes: SessionModeState | None,
    available_models: SessionModelState | None,
) -> list[ModeCategory]:
    from acp.schema import SessionConfigSelectGroup
    from agentpool.agents.modes import ModeCategory, ModeInfo

    categories: list[ModeCategory] = []

    if config_options:
        for config_opt in config_options:
            # Extract options from the config (ungrouped or grouped)
            mode_infos: list[ModeInfo] = []
            if isinstance(config_opt.options, list):
                for opt_item in config_opt.options:
                    if isinstance(opt_item, SessionConfigSelectGroup):
                        mode_infos.extend(
                            ModeInfo(
                                id=sub_opt.value,
                                name=sub_opt.name,
                                description=sub_opt.description or "",
                                category_id=config_opt.id,
                            )
                            for sub_opt in opt_item.options
                        )
                    else:
                        # Ungrouped options
                        mode_infos.append(
                            ModeInfo(
                                id=opt_item.value,
                                name=opt_item.name,
                                description=opt_item.description or "",
                                category_id=config_opt.id,
                            )
                        )

            categories.append(
                ModeCategory(
                    id=config_opt.id,
                    name=config_opt.name,
                    available_modes=mode_infos,
                    current_mode_id=config_opt.current_value,
                    category=config_opt.category or "other",
                )
            )
        return categories

    # Legacy: Convert ACP SessionModeState to ModeCategory
    if available_modes:
        acp_modes = available_modes
        modes = [
            ModeInfo(
                id=m.id,
                name=m.name,
                description=m.description or "",
                category_id="permissions",
            )
            for m in acp_modes.available_modes
        ]
        categories.append(
            ModeCategory(
                id="permissions",
                name="Mode",
                available_modes=modes,
                current_mode_id=acp_modes.current_mode_id,
                category="mode",
            )
        )

    # Legacy: Convert ACP SessionModelState to ModeCategory
    if available_models:
        acp_models = available_models
        models = [
            ModeInfo(
                id=m.model_id,
                name=m.name,
                description=m.description or "",
                category_id="model",
            )
            for m in acp_models.available_models
        ]
        categories.append(
            ModeCategory(
                id="model",
                name="Model",
                available_modes=models,
                current_mode_id=acp_models.current_model_id,
                category="model",
            )
        )

    return categories


def to_finish_reason(stop_reason: StopReason) -> FinishReason:
    return STOP_REASON_MAP.get(stop_reason, "stop")


def convert_acp_locations(
    locations: Sequence[ToolCallLocation] | None,
) -> list[LocationContentItem]:
    """Convert ACP ToolCallLocation list to native LocationContentItem list."""
    return [LocationContentItem(path=loc.path, line=loc.line) for loc in locations or []]


def convert_acp_content(content: Sequence[ToolCallContent] | None) -> list[ToolCallContentItem]:
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
                from agentpool.agents.events import TextContentItem

                result.append(TextContentItem(text=text))
    return result


def convert_to_acp_content(prompts: Sequence[UserContent]) -> list[ContentBlock]:
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

            case ImageUrl(url=url, media_type=typ):
                content_blocks.append(
                    ResourceContentBlock(uri=url, name="Image", mime_type=typ or "image/jpeg")
                )

            case AudioUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=url,
                        name="Audio",
                        mime_type=media_type or "audio/wav",
                        description="Audio content",
                    )
                )

            case DocumentUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=url,
                        name="Document",
                        mime_type=media_type or "application/pdf",
                        description="Document",
                    )
                )

            case VideoUrl(url=url, media_type=media_type):
                content_blocks.append(
                    ResourceContentBlock(
                        uri=url,
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

        # User message echo - usually ignored
        case UserMessageChunk():
            return None

        # Tool call start -> ToolCallStartEvent
        case ToolCallStart(
            tool_call_id=tool_call_id,
            title=title,
            kind=kind,
            content=content,
            locations=locations,
            raw_input=raw_input,
        ):
            return ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=title,  # ACP uses title, not separate tool_name
                title=title,
                kind=kind or "other",
                content=convert_acp_content(list(content) if content else None),
                locations=convert_acp_locations(list(locations) if locations else None),
                raw_input=raw_input or {},
            )

        # Tool call progress -> ToolCallProgressEvent or ToolCallCompleteEvent
        case ToolCallProgress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            content=content,
            raw_output=raw_output,
        ):
            # If completed, return ToolCallCompleteEvent for metadata injection
            if status == "completed":
                from agentpool.agents.events import ToolCallCompleteEvent

                return ToolCallCompleteEvent(
                    tool_call_id=tool_call_id,
                    tool_name=title or "unknown",
                    tool_input={},  # ACP doesn't provide input in progress updates
                    tool_result=str(raw_output) if raw_output else "",
                    agent_name="",  # Will be set by agent
                    message_id="",
                    metadata=None,  # Will be injected by agent from metadata accumulator
                )
            # Otherwise return progress event
            return ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                status=status or "in_progress",
                title=title,
                items=convert_acp_content(list(content) if content else None),
                message=str(raw_output) if raw_output else None,
            )

        # Plan update -> PlanUpdateEvent
        case AgentPlanUpdate(entries=entries):
            from agentpool.resource_providers.plan_provider import PlanEntry

            native_entries = [
                PlanEntry(content=e.content, priority=e.priority, status=e.status) for e in entries
            ]
            return PlanUpdateEvent(entries=native_entries)

        case _:
            return None


@overload
def mcp_config_to_acp(config: StdioMCPServerConfig) -> StdioMcpServer: ...


@overload
def mcp_config_to_acp(config: SSEMCPServerConfig) -> SseMcpServer: ...


@overload
def mcp_config_to_acp(config: StreamableHTTPMCPServerConfig) -> HttpMcpServer: ...


@overload
def mcp_config_to_acp(config: MCPServerConfig) -> McpServer: ...


def mcp_config_to_acp(config: MCPServerConfig) -> McpServer:
    """Convert native MCPServerConfig to ACP McpServer format.

    If the config has tool filtering (enabled_tools or disabled_tools),
    the server is wrapped with mcp-filter proxy to apply the filtering.

    Args:
        config: agentpool MCP server configuration

    Returns:
        ACP-compatible McpServer instance, or None if conversion not possible
    """
    from acp.schema.common import EnvVariable
    from acp.schema.mcp import HttpMcpServer, SseMcpServer, StdioMcpServer
    from agentpool_config.mcp_server import (
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    # If filtering is configured, wrap with mcp-filter first
    if config.needs_tool_filtering():
        config = config.wrap_with_mcp_filter()

    match config:
        case StdioMCPServerConfig(command=command, args=args):
            env_vars = config.get_env_vars() if hasattr(config, "get_env_vars") else {}
            env_list = [EnvVariable(name=k, value=v) for k, v in env_vars.items()]
            return StdioMcpServer(
                name=config.name or command,
                command=command,
                args=list(args) if args else [],
                env=env_list,
            )

        case SSEMCPServerConfig(url=url):
            return SseMcpServer(name=config.name or str(url), url=url, headers=[])

        case StreamableHTTPMCPServerConfig(url=url):
            return HttpMcpServer(name=config.name or str(url), url=url, headers=[])

        case _ as unreachable:
            assert_never(unreachable)


def mcp_configs_to_acp(configs: Sequence[MCPServerConfig]) -> list[McpServer]:
    """Convert a sequence of MCPServerConfig to ACP McpServer list.

    Args:
        configs: Sequence of agentpool MCP server configurations

    Returns:
        List of ACP-compatible McpServer instances (skips unconvertible configs)
    """
    return [mcp_config_to_acp(config) for config in configs]
