"""Convert between Codex and AgentPool types.

Provides converters for:
- Event conversion (Codex streaming events -> AgentPool events)
- MCP server configs (Native configs -> Codex types)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

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
    from collections.abc import Sequence

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


def _format_tool_result(item: ThreadItem) -> str:
    """Format tool result from a completed ThreadItem.

    Args:
        item: Completed thread item

    Returns:
        Formatted result string
    """
    from codex_adapter.models import ThreadItemCommandExecution, ThreadItemMcpToolCall

    match item:
        case ThreadItemCommandExecution():
            return item.aggregated_output or ""
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


def codex_to_native_event(event: CodexEvent) -> RichAgentStreamEvent[str] | None:  # noqa: PLR0911
    """Convert Codex streaming event to native AgentPool event.

    Args:
        event: Codex event from app-server

    Returns:
        Native event or None if not convertible
    """
    from agentpool.agents.events import (
        CompactionEvent,
        PartDeltaEvent,
        PlanUpdateEvent,
        ToolCallCompleteEvent,
        ToolCallProgressEvent,
        ToolCallStartEvent,
    )
    from agentpool.resource_providers.plan_provider import PlanEntry
    from codex_adapter.models import (
        AgentMessageDeltaData,
        CommandExecutionOutputDeltaData,
        ItemCompletedData,
        ItemStartedData,
        McpToolCallProgressData,
        ReasoningTextDeltaData,
        ThreadCompactedData,
        ThreadItemCommandExecution,
        ThreadItemMcpToolCall,
        TurnPlanUpdatedData,
    )

    match event.event_type:
        # Text deltas from agent messages
        case "item/agentMessage/delta" if isinstance(event.data, AgentMessageDeltaData):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=event.data.delta))

        # Reasoning/thinking deltas
        case "item/reasoning/textDelta" if isinstance(event.data, ReasoningTextDeltaData):
            return PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=event.data.delta))

        # Tool/command started
        case "item/started" if isinstance(event.data, ItemStartedData):
            item = event.data.item
            if part := _thread_item_to_tool_call_part(item):
                # Extract title based on tool type
                match item:
                    case ThreadItemCommandExecution():
                        title = f"Execute: {item.command}"
                    case ThreadItemMcpToolCall():
                        title = f"Call {item.tool}"
                    case _:
                        title = f"Call {part.tool_name}"

                return ToolCallStartEvent(
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                    title=title,
                    raw_input=part.args_as_dict(),
                )

        # Tool/command completed
        case "item/completed" if isinstance(event.data, ItemCompletedData):
            item = event.data.item
            if part := _thread_item_to_tool_call_part(item):
                return ToolCallCompleteEvent(
                    tool_name=part.tool_name,
                    tool_call_id=part.tool_call_id,
                    tool_input=part.args_as_dict(),
                    tool_result=_format_tool_result(item),
                    agent_name="codex",  # Will be overridden by agent
                    message_id=event.data.turn_id,
                )

        # Command execution output streaming
        case "item/commandExecution/outputDelta" if isinstance(
            event.data, CommandExecutionOutputDeltaData
        ):
            # This is streaming tool output - emit as text delta
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=event.data.delta))

        # MCP tool call progress
        case "item/mcpToolCall/progress" if isinstance(event.data, McpToolCallProgressData):
            return ToolCallProgressEvent(
                tool_call_id=event.data.item_id,
                message=event.data.message,
            )

        # Thread compacted - history was summarized
        case "thread/compacted" if isinstance(event.data, ThreadCompactedData):
            return CompactionEvent(session_id=event.data.thread_id, phase="completed")

        # Turn plan updated - agent's plan for current turn
        case "turn/plan/updated" if isinstance(event.data, TurnPlanUpdatedData):
            # Convert Codex steps to PlanEntry format
            entries = [
                PlanEntry(
                    content=step.step,
                    priority="medium",  # Codex doesn't provide priority
                    status="in_progress" if step.status == "inProgress" else step.status,
                )
                for step in event.data.plan
            ]
            return PlanUpdateEvent(entries=entries)

    return None


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
