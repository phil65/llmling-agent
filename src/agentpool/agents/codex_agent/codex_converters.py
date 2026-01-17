"""Convert between Codex and AgentPool types.

Provides converters for:
- Event conversion (Codex streaming events -> AgentPool events)
- MCP server configs (Native configs -> Codex types)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool_config.mcp_server import (
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )
    from codex_adapter.codex_types import HttpMcpServer, McpServerConfig, StdioMcpServer
    from codex_adapter.events import CodexEvent
    from codex_adapter.models import ThreadItem


@overload
def mcp_config_to_codex(config: StdioMCPServerConfig) -> tuple[str, StdioMcpServer]: ...


@overload
def mcp_config_to_codex(config: SSEMCPServerConfig) -> tuple[str, HttpMcpServer]: ...


@overload
def mcp_config_to_codex(
    config: StreamableHTTPMCPServerConfig,
) -> tuple[str, HttpMcpServer]: ...


@overload
def mcp_config_to_codex(config: MCPServerConfig) -> tuple[str, McpServerConfig]: ...


def mcp_config_to_codex(config: MCPServerConfig) -> tuple[str, McpServerConfig]:
    """Convert native MCPServerConfig to (name, Codex McpServerConfig) tuple.

    Args:
        config: Native MCP server configuration

    Returns:
        Tuple of (server name, Codex-compatible MCP server configuration)
    """
    from agentpool_config.mcp_server import (
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )
    from codex_adapter.codex_types import HttpMcpServer, StdioMcpServer

    # Name should not be None by the time we use it
    server_name = config.name or f"server_{id(config)}"
    match config:
        case StdioMCPServerConfig():
            return (
                server_name,
                StdioMcpServer(
                    command=config.command,
                    args=config.args or [],
                    env=config.env,
                    enabled=config.enabled,
                ),
            )

        case SSEMCPServerConfig():
            # Codex uses HTTP transport for SSE
            # SSE config just has URL, no separate auth fields
            return (server_name, HttpMcpServer(url=str(config.url), enabled=config.enabled))

        case StreamableHTTPMCPServerConfig():
            # StreamableHTTP has headers field
            headers = config.headers if config.headers else None
            return (
                server_name,
                HttpMcpServer(url=str(config.url), http_headers=headers, enabled=config.enabled),
            )

        case _:
            msg = f"Unsupported MCP server config type: {type(config)}"
            raise TypeError(msg)


def mcp_configs_to_codex(
    configs: Sequence[MCPServerConfig],
) -> list[tuple[str, McpServerConfig]]:
    """Convert a sequence of MCPServerConfig to list of (name, config) tuples."""
    return [mcp_config_to_codex(c) for c in configs]


def _format_tool_result(item: ThreadItem) -> str:  # noqa: PLR0911
    """Format tool result from a completed ThreadItem.

    Args:
        item: Completed thread item

    Returns:
        Formatted result string
    """
    from codex_adapter.models import (
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemMcpToolCall,
    )

    match item:
        case ThreadItemCommandExecution():
            output = item.aggregated_output or ""
            if output:
                return f"```\n{output}\n```"
            return ""
        case ThreadItemFileChange():
            # Format file changes with their diffs
            parts = []
            for change in item.changes:
                kind = change.kind.kind  # "add", "delete", or "update"
                path = change.path
                parts.append(f"{kind.upper()}: {path}")
                if change.diff:
                    parts.append(change.diff)
            return "\n".join(parts)
        case ThreadItemMcpToolCall():
            if item.result and item.result.content:
                texts = [str(block.model_dump().get("text", "")) for block in item.result.content]
                return "\n".join(texts)
            if item.error:
                return f"Error: {item.error.message}"
            return ""
        case _:
            return ""


def _thread_item_to_tool_return_part(  # noqa: PLR0911
    item: ThreadItem,
) -> ToolReturnPart | BuiltinToolReturnPart | None:
    """Convert a completed ThreadItem to a ToolReturnPart or BuiltinToolReturnPart.

    Codex built-in tools (bash, file changes, web search, etc.) are converted to
    BuiltinToolReturnPart since they're provided by the remote Codex agent.
    MCP tools are converted to ToolReturnPart (they may be from local ToolBridge).

    Args:
        item: Completed thread item from Codex

    Returns:
        ToolReturnPart for MCP tools, BuiltinToolReturnPart for Codex built-ins, or None
    """
    from codex_adapter.models import (
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemImageView,
        ThreadItemMcpToolCall,
        ThreadItemWebSearch,
    )

    # Only process completed items
    if hasattr(item, "status") and item.status != "completed":
        return None

    result = _format_tool_result(item)

    match item:
        case ThreadItemCommandExecution():
            return BuiltinToolReturnPart(
                tool_name="bash",
                content=result,
                tool_call_id=item.id,
            )
        case ThreadItemFileChange():
            return BuiltinToolReturnPart(
                tool_name="file_change",
                content=result,
                tool_call_id=item.id,
            )
        case ThreadItemWebSearch():
            return BuiltinToolReturnPart(
                tool_name="web_search",
                content=result,
                tool_call_id=item.id,
            )
        case ThreadItemImageView():
            return BuiltinToolReturnPart(
                tool_name="image_view",
                content=result,
                tool_call_id=item.id,
            )
        case ThreadItemMcpToolCall():
            # TODO: Distinguish between local (ToolBridge) and remote MCP tools
            # See matching TODO in _thread_item_to_tool_call_part
            return ToolReturnPart(
                tool_name=item.tool,
                content=result,
                tool_call_id=item.id,
            )
        case _:
            return None


def _thread_item_to_tool_call_part(
    item: ThreadItem,
) -> ToolCallPart | BuiltinToolCallPart | None:
    """Convert a ThreadItem to a ToolCallPart or BuiltinToolCallPart.

    Codex built-in tools (bash, file changes, web search, etc.) are converted to
    BuiltinToolCallPart since they're provided by the remote Codex agent.
    MCP tools are converted to ToolCallPart (they may be from local ToolBridge).

    Args:
        item: Thread item from Codex

    Returns:
        ToolCallPart for MCP tools, BuiltinToolCallPart for Codex built-ins, or None
    """
    from codex_adapter.models import (
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemImageView,
        ThreadItemMcpToolCall,
        ThreadItemWebSearch,
    )

    match item:
        case ThreadItemCommandExecution():
            return BuiltinToolCallPart(
                tool_name="bash",
                args={"command": item.command, "cwd": item.cwd},
                tool_call_id=item.id,
            )
        case ThreadItemFileChange():
            return BuiltinToolCallPart(
                tool_name="file_change",
                args={"changes": [c.model_dump() for c in item.changes]},
                tool_call_id=item.id,
            )
        case ThreadItemWebSearch():
            return BuiltinToolCallPart(
                tool_name="web_search",
                args={"query": item.query},
                tool_call_id=item.id,
            )
        case ThreadItemImageView():
            return BuiltinToolCallPart(
                tool_name="image_view",
                args={"path": item.path},
                tool_call_id=item.id,
            )
        case ThreadItemMcpToolCall():
            # TODO: Distinguish between local (ToolBridge) and remote MCP tools
            # Currently all MCP tools use ToolCallPart, but ideally:
            # - Tools from AgentPool's ToolBridge → ToolCallPart (our tools)
            # - Tools from Codex's own MCP servers → BuiltinToolCallPart (their tools)
            # This requires tracking which tools came from ToolBridge vs Codex config
            args = item.arguments if isinstance(item.arguments, dict) else {"args": item.arguments}
            return ToolCallPart(
                tool_name=item.tool,
                args=args,
                tool_call_id=item.id,
            )
        case _:
            return None


async def convert_codex_stream(  # noqa: PLR0915
    events: AsyncIterator[CodexEvent],
) -> AsyncIterator[RichAgentStreamEvent[Any]]:
    """Convert Codex event stream to native events with stateful accumulation.

    This async generator handles stateful conversion of Codex events, including:
    - Accumulating command execution output deltas into ToolCallProgressEvents
    - Accumulating file change output deltas into ToolCallProgressEvents
    - Simple 1:1 conversion for other event types

    Args:
        events: Async iterator of Codex events from the app-server

    Yields:
        Native AgentPool stream events
    """
    from agentpool.agents.events import (
        CompactionEvent,
        PartDeltaEvent,
        PlanUpdateEvent,
        TextContentItem,
        ToolCallCompleteEvent,
        ToolCallProgressEvent,
        ToolCallStartEvent,
    )
    from agentpool.resource_providers.plan_provider import PlanEntry
    from codex_adapter.models import (
        AgentMessageDeltaData,
        CommandExecutionOutputDeltaData,
        FileChangeOutputDeltaData,
        ItemCompletedData,
        ItemStartedData,
        McpToolCallProgressData,
        ReasoningTextDeltaData,
        ThreadCompactedData,
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemMcpToolCall,
        TurnPlanUpdatedData,
    )

    # Accumulation state for streaming tool outputs
    tool_outputs: dict[str, list[str]] = {}

    async for event in events:
        match event.event_type:
            # === Stateful: Accumulate command execution output ===
            case "item/commandExecution/outputDelta" if isinstance(
                event.data, CommandExecutionOutputDeltaData
            ):
                item_id = event.data.item_id
                if item_id not in tool_outputs:
                    tool_outputs[item_id] = []
                tool_outputs[item_id].append(event.data.delta)

                # Emit accumulated progress with replace semantics, wrapped in code block
                output = "".join(tool_outputs[item_id])
                yield ToolCallProgressEvent(
                    tool_call_id=item_id,
                    items=[TextContentItem(text=f"```\n{output}\n```")],
                    replace_content=True,
                )

            # === File change output delta - ignore the summary, we show diff from item/started ===
            case "item/fileChange/outputDelta" if isinstance(event.data, FileChangeOutputDeltaData):
                # The outputDelta is just "Success. Updated..." summary - not useful
                # We already emitted the actual diff content in item/started
                pass

            # === Stateless: Text deltas from agent messages ===
            case "item/agentMessage/delta" if isinstance(event.data, AgentMessageDeltaData):
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=event.data.delta))

            # === Stateless: Reasoning/thinking deltas ===
            case "item/reasoning/textDelta" if isinstance(event.data, ReasoningTextDeltaData):
                yield PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=event.data.delta)
                )

            # === Stateless: Tool/command started ===
            case "item/started" if isinstance(event.data, ItemStartedData):
                item = event.data.item
                if part := _thread_item_to_tool_call_part(item):
                    # Extract title based on tool type
                    match item:
                        case ThreadItemCommandExecution():
                            title = f"Execute: {item.command}"
                        case ThreadItemFileChange():
                            # Build title from file paths
                            paths = [c.path for c in item.changes[:3]]  # First 3 paths
                            if len(item.changes) > 3:  # noqa: PLR2004
                                title = f"Edit: {', '.join(paths)} (+{len(item.changes) - 3} more)"
                            else:
                                title = f"Edit: {', '.join(paths)}"
                        case ThreadItemMcpToolCall():
                            title = f"Call {item.tool}"
                        case _:
                            title = f"Call {part.tool_name}"

                    yield ToolCallStartEvent(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        title=title,
                        raw_input=part.args_as_dict(),
                    )

                    # For file changes, immediately emit the diff as progress
                    if isinstance(item, ThreadItemFileChange):
                        diff_parts = []
                        for change in item.changes:
                            kind = change.kind.kind
                            path = change.path
                            diff_parts.append(f"{kind.upper()}: {path}")
                            if change.diff:
                                diff_parts.append(change.diff)
                        if diff_parts:
                            yield ToolCallProgressEvent(
                                tool_call_id=part.tool_call_id,
                                items=[TextContentItem(text="\n".join(diff_parts))],
                            )

            # === Stateful: Tool/command completed - clean up accumulator ===
            case "item/completed" if isinstance(event.data, ItemCompletedData):
                item = event.data.item
                # Clean up accumulated output for this item
                if hasattr(item, "id") and item.id in tool_outputs:
                    del tool_outputs[item.id]

                if part := _thread_item_to_tool_call_part(item):
                    yield ToolCallCompleteEvent(
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        tool_input=part.args_as_dict(),
                        tool_result=_format_tool_result(item),
                        agent_name="codex",  # Will be overridden by agent
                        message_id=event.data.turn_id,
                    )

            # === Stateless: MCP tool call progress ===
            case "item/mcpToolCall/progress" if isinstance(event.data, McpToolCallProgressData):
                yield ToolCallProgressEvent(
                    tool_call_id=event.data.item_id,
                    message=event.data.message,
                )

            # === Stateless: Thread compacted ===
            case "thread/compacted" if isinstance(event.data, ThreadCompactedData):
                yield CompactionEvent(session_id=event.data.thread_id, phase="completed")

            # === Stateless: Turn plan updated ===
            case "turn/plan/updated" if isinstance(event.data, TurnPlanUpdatedData):
                entries = [
                    PlanEntry(
                        content=step.step,
                        priority="medium",  # Codex doesn't provide priority
                        status="in_progress" if step.status == "inProgress" else step.status,
                    )
                    for step in event.data.plan
                ]
                yield PlanUpdateEvent(entries=entries)

            # Ignore other events (token usage, turn started/completed, etc.)
            case _:
                pass


def event_to_part(
    event: CodexEvent,
) -> (
    TextPart
    | ThinkingPart
    | ToolCallPart
    | BuiltinToolCallPart
    | ToolReturnPart
    | BuiltinToolReturnPart
    | None
):
    """Convert Codex event to part for message construction.

    This is for building final messages, not for streaming.

    Handles both tool calls (item/started) and tool returns (item/completed).

    Args:
        event: Codex event

    Returns:
        Part or None
    """
    from codex_adapter.models import (
        AgentMessageDeltaData,
        ItemCompletedData,
        ItemStartedData,
        ReasoningTextDeltaData,
    )

    match event.event_type:
        case "item/agentMessage/delta":
            if isinstance(event.data, AgentMessageDeltaData):
                return TextPart(content=event.data.delta)

        case "item/reasoning/textDelta":
            if isinstance(event.data, ReasoningTextDeltaData):
                return ThinkingPart(content=event.data.delta)

        case "item/started":
            if isinstance(event.data, ItemStartedData):
                return _thread_item_to_tool_call_part(event.data.item)

        case "item/completed":
            if isinstance(event.data, ItemCompletedData):
                return _thread_item_to_tool_return_part(event.data.item)

    return None
