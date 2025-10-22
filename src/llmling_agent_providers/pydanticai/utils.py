"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload
from uuid import uuid4

from pydantic_ai import messages as _messages
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from llmling_agent.messaging.messages import ChatMessage
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
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequestPart,
        ModelResponsePart,
        UserContent as PydanticUserContent,
    )

    from llmling_agent.common_types import MessageRole
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


def convert_model_message(  # noqa: PLR0911
    message: ModelMessage | ModelRequestPart | ModelResponsePart,
    tools: dict[str, Tool],
    agent_name: str,
    filter_system_prompts: bool = False,
) -> ChatMessage:
    """Convert a pydantic-ai message to our ChatMessage format.

    Also supports converting parts of a message (with limitations then of course)

    Args:
        message: Message to convert (ModelMessage or its parts)
        tools: Original Tool set to enrich ToolCallInfos with additional info
        agent_name: Name of the agent of this message
        filter_system_prompts: Whether to filter out system prompt parts from
                               ModelRequests

    Returns:
        Converted ChatMessage

    Raises:
        ValueError: If message type is not supported
    """
    match message:
        case ModelRequest():
            # Collect content from all parts
            content_parts = []
            role: MessageRole = "system"
            for part in message.parts:
                match part:
                    case UserPromptPart():
                        content_parts.append(str(part.content))
                        role = "user"
                    case SystemPromptPart() if not filter_system_prompts:
                        content_parts.append(str(part.content))
            return ChatMessage(content="\n".join(content_parts), role=role)

        case ModelResponse():
            # Collect content and tool calls from all parts
            tool_calls = get_tool_calls([message], tools, None)
            parts = [format_part(p) for p in message.parts if isinstance(p, TextPart)]
            content = "\n".join(parts)
            return ChatMessage(content=content, role="assistant", tool_calls=tool_calls)

        case TextPart() as part:
            return ChatMessage(content=format_part(part), role="assistant")

        case UserPromptPart() as part:
            return ChatMessage(content=format_part(part), role="user")

        case SystemPromptPart() as part:
            return ChatMessage(content=format_part(part), role="system")

        case ToolCallPart():
            args = message.args_as_dict()
            info = ToolCallInfo(
                tool_name=message.tool_name,
                args=args,
                agent_name=agent_name,
                result=None,  # Not available yet
                tool_call_id=message.tool_call_id or str(uuid4()),
            )
            content = f"Tool call: {message.tool_name}\nArgs: {args}"
            return ChatMessage(content=content, role="assistant", tool_calls=[info])

        case ToolReturnPart():
            info = ToolCallInfo(
                tool_name=message.tool_name,
                agent_name=agent_name,
                args={},  # No args in return part
                result=message.content,
                tool_call_id=message.tool_call_id or str(uuid4()),
                timestamp=message.timestamp,
            )
            content = f"Tool {message.tool_name} returned: {message.content}"
            return ChatMessage(content=content, role="assistant", tool_calls=[info])

        case RetryPromptPart():
            error_content = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(f"- {err['loc']}: {err['msg']}" for err in message.content)
            )
            return ChatMessage(content=f"Retry needed: {error_content}", role="assistant")

        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise ValueError(msg)


def to_model_request(message: ChatMessage[str | Content]) -> ModelRequest:
    """Convert ChatMessage to pydantic-ai ModelMessage."""
    import anyenv

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
) -> list[str | PydanticUserContent]:
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
    result: list[str | PydanticUserContent] = []
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
