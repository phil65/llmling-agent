"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload
from uuid import uuid4

import anyenv
from pydantic_ai import messages as _messages
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from llmling_agent.models.content import BaseContent
from llmling_agent.tools import ToolCallInfo


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.mcp import (
        MCPServer,
        MCPServerSSE,
        MCPServerStdio,
        MCPServerStreamableHTTP,
    )
    from pydantic_ai.messages import ModelMessage, UserContent

    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.content import Content
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import (
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )


def format_part(  # noqa: PLR0911
    response: str | _messages.ModelRequestPart | _messages.ModelResponsePart,
) -> str:
    """Format any kind of response in a readable way.

    Args:
        response: Response part to format

    Returns:
        A human-readable string representation
    """
    match response:
        case str():
            return response
        case _messages.TextPart():
            return response.content
        case _messages.ToolCallPart():
            args = str(response.args)
            return f"Tool call: {response.tool_name}\nArgs: {args}"
        case _messages.ToolReturnPart():
            return f"Tool {response.tool_name} returned: {response.content}"
        case _messages.RetryPromptPart():
            if isinstance(response.content, str):
                return f"Retry needed: {response.content}"
            return f"Validation errors:\n{response.content}"
        case _:
            return str(response)


def get_tool_calls(
    messages: list[ModelMessage],
    tools: dict[str, Tool] | None = None,
    agent_name: str | None = None,
    context_data: Any | None = None,
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        tools: Original Tool set to enrich ToolCallInfos with additional info
        agent_name: Name of the caller
        context_data: Optional context data to attach to tool calls
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
            tools.get(part.tool_name),
            agent_name=agent_name,
            context_data=context_data,
        )
        for part in parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id in call_parts
    ]


def parts_to_tool_call_info(
    call_part: ToolCallPart,
    return_part: ToolReturnPart,
    tool_info: Tool | None,
    agent_name: str | None = None,
    context_data: Any | None = None,
) -> ToolCallInfo:
    """Convert matching tool call and return parts into a ToolCallInfo."""
    return ToolCallInfo(
        tool_name=call_part.tool_name,
        args=call_part.args_as_dict(),
        agent_name=agent_name or "UNSET",
        result=return_part.content,
        tool_call_id=call_part.tool_call_id or str(uuid4()),
        timestamp=return_part.timestamp,
        context_data=context_data,
        agent_tool_name=tool_info.agent_name if tool_info else None,
    )


def to_model_request(message: ChatMessage[str | Content]) -> ModelRequest:
    """Convert ChatMessage to pydantic-ai ModelMessage."""
    match message.content:
        case BaseContent():
            content = [message.content.to_openai_format()]
            part = UserPromptPart(content=anyenv.dump_json({"content": content}))
            return ModelRequest(parts=[part])
        case str():
            part_cls = {
                "user": UserPromptPart,
                "system": SystemPromptPart,
                "assistant": UserPromptPart,
            }.get(message.role)
            if not part_cls:
                msg = f"Unknown message role: {message.role}"
                raise ValueError(msg)
            return ModelRequest(parts=[part_cls(content=message.content)])


async def convert_prompts_to_user_content(
    prompts: Sequence[str | Content],
) -> list[str | UserContent]:
    """Convert our prompts to pydantic-ai compatible format.

    Args:
        prompts: Sequence of string prompts or Content objects

    Returns:
        List of strings and pydantic-ai UserContent objects
    """
    from llmling_agent.prompts.convert import format_prompts
    from llmling_agent_providers.pydanticai.convert_content import content_to_pydantic_ai

    # Special case: if we only have string prompts, format them together
    # if all(isinstance(p, str) for p in prompts):
    #     formatted = await format_prompts(prompts)
    #     return [formatted]

    # Otherwise, process each item individually in order
    result: list[str | UserContent] = []
    for p in prompts:
        if isinstance(p, str):
            formatted = await format_prompts([p])
            result.append(formatted)
        elif p_content := content_to_pydantic_ai(p):
            result.append(p_content)

    return result


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
