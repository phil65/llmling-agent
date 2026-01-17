"""Convert between Codex and AgentPool types.

Provides converters for:
- Event conversion (Codex streaming events -> AgentPool events)
- MCP server configs (Native configs -> Codex types)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from pydantic_ai import TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta, ToolCallPart


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
        ToolCallStartEvent,
    )
    from agentpool.resource_providers.plan_provider import PlanEntry
    from codex_adapter.models import (
        AgentMessageDeltaData,
        CommandExecutionOutputDeltaData,
        ItemCompletedData,
        ItemStartedData,
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
            # Command execution (built-in shell commands)
            if isinstance(item, ThreadItemCommandExecution):
                return ToolCallStartEvent(
                    tool_call_id=item.id,
                    tool_name="bash",
                    title=f"Execute: {item.command}",
                    raw_input={"command": item.command, "cwd": item.cwd},
                )
            # MCP tool calls
            if isinstance(item, ThreadItemMcpToolCall):
                return ToolCallStartEvent(
                    tool_call_id=item.id,
                    tool_name=item.tool,
                    title=f"Call {item.tool}",
                    raw_input=(
                        item.arguments
                        if isinstance(item.arguments, dict)
                        else {"args": item.arguments}
                    ),
                )

        # Tool/command completed
        case "item/completed" if isinstance(event.data, ItemCompletedData):
            item = event.data.item
            # Command execution completed
            if isinstance(item, ThreadItemCommandExecution):
                output = item.aggregated_output or ""
                return ToolCallCompleteEvent(
                    tool_name="bash",
                    tool_call_id=item.id,
                    tool_input={"command": item.command, "cwd": item.cwd},
                    tool_result=output,
                    agent_name="codex",  # Will be overridden by agent
                    message_id=event.data.turn_id,
                )
            # MCP tool call completed
            if isinstance(item, ThreadItemMcpToolCall):
                result_text = ""  # Format result content as string
                if item.result and item.result.content:
                    # McpContentBlock allows extra fields, cast to dict to access
                    texts = [str(i.model_dump().get("text", "")) for i in item.result.content]
                    result_text = "\n".join(texts)
                elif item.error:
                    result_text = f"Error: {item.error.message}"
                tool_input = args if isinstance((args := item.arguments), dict) else {"args": args}
                return ToolCallCompleteEvent(
                    tool_name=item.tool,
                    tool_call_id=item.id,
                    tool_input=tool_input,
                    tool_result=result_text,
                    agent_name="codex",  # Will be overridden by agent
                    message_id=event.data.turn_id,
                )

        # Command execution output streaming
        case "item/commandExecution/outputDelta" if isinstance(
            event.data, CommandExecutionOutputDeltaData
        ):
            # This is streaming tool output - emit as text delta
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=event.data.delta))
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


def event_to_part(event: CodexEvent) -> TextPart | ThinkingPart | ToolCallPart | None:
    """Convert Codex event to part for message construction.

    This is for building final messages, not for streaming.

    Args:
        event: Codex event

    Returns:
        Part or None
    """
    from codex_adapter.models import (
        AgentMessageDeltaData,
        ItemStartedData,
        ReasoningTextDeltaData,
        ThreadItemCommandExecution,
        ThreadItemMcpToolCall,
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
                item = event.data.item
                if isinstance(item, ThreadItemCommandExecution):
                    return ToolCallPart(
                        tool_name="bash",
                        args={"command": item.command, "cwd": item.cwd},
                        tool_call_id=item.id,
                    )
                if isinstance(item, ThreadItemMcpToolCall):
                    return ToolCallPart(
                        tool_name=item.tool,
                        args=(
                            item.arguments
                            if isinstance(item.arguments, dict)
                            else {"args": item.arguments}
                        ),
                        tool_call_id=item.id,
                    )

    return None
