"""Claude Agent SDK to native event converters.

This module provides conversion from Claude Agent SDK message types to native
agentpool streaming events, enabling ClaudeCodeAgent to yield the same
event types as native agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import PartDeltaEvent, TextPartDelta, ThinkingPartDelta

from agentpool.agents.events import ToolCallCompleteEvent, ToolCallStartEvent


if TYPE_CHECKING:
    from claude_agent_sdk import ContentBlock, McpServerConfig, Message, ToolUseBlock

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool_config.mcp_server import MCPServerConfig as NativeMCPServerConfig


def content_block_to_event(block: ContentBlock, index: int = 0) -> RichAgentStreamEvent[Any] | None:
    """Convert a Claude SDK ContentBlock to a streaming event.

    Args:
        block: Claude SDK content block
        index: Part index for the event

    Returns:
        Corresponding streaming event, or None if not mappable
    """
    from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock

    from agentpool.agents.events.infer_info import derive_rich_tool_info

    match block:
        case TextBlock(text=text):
            return PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=text))
        case ThinkingBlock(thinking=thinking):
            return PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=thinking))
        case ToolUseBlock(id=tool_id, name=name, input=input_data):
            rich_info = derive_rich_tool_info(name, input_data)
            return ToolCallStartEvent(
                tool_call_id=tool_id,
                tool_name=name,
                title=rich_info.title,
                kind=rich_info.kind,
                locations=rich_info.locations,
                content=rich_info.content,
                raw_input=input_data,
            )
        case _:
            return None


def claude_message_to_events(
    message: Message,
    agent_name: str = "",
    pending_tool_calls: dict[str, ToolUseBlock] | None = None,
) -> list[RichAgentStreamEvent[Any]]:
    """Convert a Claude SDK Message to a list of streaming events.

    Args:
        message: Claude SDK message (UserMessage, AssistantMessage, etc.)
        agent_name: Name of the agent for event attribution
        pending_tool_calls: Dict to track tool calls awaiting results

    Returns:
        List of corresponding streaming events
    """
    from claude_agent_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock

    events: list[RichAgentStreamEvent[Any]] = []

    match message:
        case AssistantMessage(content=content):
            for idx, block in enumerate(content):
                # Track tool use blocks for later pairing with results
                if isinstance(block, ToolUseBlock) and pending_tool_calls is not None:
                    pending_tool_calls[block.id] = block

                # Handle tool results - pair with pending tool call
                if isinstance(block, ToolResultBlock) and pending_tool_calls is not None:
                    tool_use = pending_tool_calls.pop(block.tool_use_id, None)
                    if tool_use:
                        complete_event = ToolCallCompleteEvent(
                            tool_name=tool_use.name,
                            tool_call_id=block.tool_use_id,
                            tool_input=tool_use.input,
                            tool_result=block.content,
                            agent_name=agent_name,
                            message_id="",
                        )
                        events.append(complete_event)
                    continue

                # Convert other blocks to events
                if event := content_block_to_event(block, index=idx):
                    events.append(event)

        case _:
            # UserMessage, SystemMessage, ResultMessage - no events to emit
            pass

    return events


def convert_mcp_servers_to_sdk_format(
    mcp_servers: list[NativeMCPServerConfig],
) -> dict[str, McpServerConfig]:
    """Convert internal MCPServerConfig to Claude SDK format.

    Returns:
        Dict mapping server names to SDK-compatible config dicts
    """
    from claude_agent_sdk import McpServerConfig

    from agentpool_config.mcp_server import (
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    result: dict[str, McpServerConfig] = {}

    for idx, server in enumerate(mcp_servers):
        # Determine server name
        if server.name:
            name = server.name
        elif isinstance(server, StdioMCPServerConfig) and server.args:
            name = server.args[-1].split("/")[-1].split("@")[0]
        elif isinstance(server, StdioMCPServerConfig):
            name = server.command
        elif isinstance(server, SSEMCPServerConfig | StreamableHTTPMCPServerConfig):
            from urllib.parse import urlparse

            name = urlparse(str(server.url)).hostname or f"server_{idx}"
        else:
            name = f"server_{idx}"

        # Build SDK-compatible config
        config: dict[str, Any]
        match server:
            case StdioMCPServerConfig(command=command, args=args):
                config = {"type": "stdio", "command": command, "args": args}
                if server.env:
                    config["env"] = server.get_env_vars()
            case SSEMCPServerConfig(url=url):
                config = {"type": "sse", "url": str(url)}
                if server.headers:
                    config["headers"] = server.headers
            case StreamableHTTPMCPServerConfig(url=url):
                config = {"type": "http", "url": str(url)}
                if server.headers:
                    config["headers"] = server.headers

        result[name] = cast(McpServerConfig, config)

    return result


def to_output_format(output_type: type) -> dict[str, Any] | None:
    """Convert to SDK output format dict."""
    from pydantic import TypeAdapter

    # Build structured output format if needed
    output_format: dict[str, Any] | None = None
    if output_type is not str:
        adapter = TypeAdapter[Any](output_type)
        schema = adapter.json_schema()
        output_format = {"type": "json_schema", "schema": schema}
    return output_format
