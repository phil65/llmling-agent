"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload
from uuid import uuid4

from pydantic_ai import ToolCallPart, ToolReturnPart

from llmling_agent.tools import ToolCallInfo


if TYPE_CHECKING:
    from pydantic_ai import ModelMessage
    from pydantic_ai.mcp import (
        MCPServer,
        MCPServerSSE,
        MCPServerStdio,
        MCPServerStreamableHTTP,
    )

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import (
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )


def get_tool_calls(
    messages: list[ModelMessage],
    tools: dict[str, Tool] | None = None,
    agent_name: str | None = None,
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        tools: Original Tool set to enrich ToolCallInfos with additional info
        agent_name: Name of the caller
    """
    tools = tools or {}
    parts = [part for message in messages for part in message.parts]
    call_parts = {
        part.tool_call_id: part
        for part in parts
        if isinstance(part, ToolCallPart) and part.tool_call_id
    }
    return [
        parts_to_tool_call_info(
            call_parts[part.tool_call_id],
            part,
            t.agent_name if (t := tools.get(part.tool_name)) else None,
            agent_name=agent_name,
        )
        for part in parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id in call_parts
    ]


def parts_to_tool_call_info(
    call_part: ToolCallPart,
    return_part: ToolReturnPart,
    agent_tool_name: str | None,
    agent_name: str | None = None,
) -> ToolCallInfo:
    """Convert matching tool call and return parts into a ToolCallInfo."""
    return ToolCallInfo(
        tool_name=call_part.tool_name,
        args=call_part.args_as_dict(),
        agent_name=agent_name or "UNSET",
        result=return_part.content,
        tool_call_id=call_part.tool_call_id or str(uuid4()),
        timestamp=return_part.timestamp,
        agent_tool_name=agent_tool_name,
    )


@overload
def mcp_config_to_pydantic_ai(config: StdioMCPServerConfig) -> MCPServerStdio: ...


@overload
def mcp_config_to_pydantic_ai(config: SSEMCPServerConfig) -> MCPServerSSE: ...


@overload
def mcp_config_to_pydantic_ai(
    config: StreamableHTTPMCPServerConfig,
) -> MCPServerStreamableHTTP: ...


@overload
def mcp_config_to_pydantic_ai(config: MCPServerConfig) -> MCPServer: ...


def mcp_config_to_pydantic_ai(config: MCPServerConfig) -> MCPServer:
    """Convert llmling-agent MCP server config to pydantic-ai MCP server.

    Args:
        config: The MCP server configuration to convert

    Returns:
        A pydantic-ai MCP server instance

    Raises:
        ValueError: If server type is not supported
    """
    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

    match config.type:
        case "stdio":
            return MCPServerStdio(
                command=config.command,
                args=config.args,
                env=config.get_env_vars() if config.env else None,
                id=config.name,
                timeout=config.timeout,
            )

        case "sse":
            return MCPServerSSE(
                url=str(config.url),
                headers=config.headers,
                id=config.name,
                timeout=config.timeout,
            )

        case "streamable-http":
            return MCPServerStreamableHTTP(
                url=str(config.url),
                headers=config.headers,
                id=config.name,
                timeout=config.timeout,
            )

        case _:
            msg = f"Unsupported MCP server type: {config.type}"
            raise ValueError(msg)
