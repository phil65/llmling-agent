"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic_ai import messages as _messages
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.content import (
    BaseContent,
    Content,
)
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from llmling_agent.common_types import MessageRole
    from llmling_agent.tools.base import ToolInfo


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
            if isinstance(response.args, _messages.ArgsJson):
                args = response.args.args_json
            else:
                args = str(response.args.args_dict)
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
    tools: dict[str, ToolInfo] | None = None,
    agent_name: str | None = None,
    context_data: Any | None = None,
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        tools: Original ToolInfo set to enrich ToolCallInfos with additional info
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
    tool_info: ToolInfo | None,
    agent_name: str | None = None,
    context_data: Any | None = None,
) -> ToolCallInfo:
    """Convert matching tool call and return parts into a ToolCallInfo."""
    args = (
        call_part.args.args_dict
        if isinstance(call_part.args, ArgsDict)
        else json.loads(call_part.args.args_json)
    )

    return ToolCallInfo(
        tool_name=call_part.tool_name,
        args=args,
        agent_name=agent_name or "UNSET",
        result=return_part.content,
        tool_call_id=call_part.tool_call_id or str(uuid4()),
        timestamp=return_part.timestamp,
        context_data=context_data,
        agent_tool_name=tool_info.agent_name if tool_info else None,
    )


def convert_model_message(
    message: ModelMessage | ModelRequestPart | ModelResponsePart,
    tools: dict[str, ToolInfo],
    agent_name: str,
) -> ChatMessage:
    """Convert a pydantic-ai message to our ChatMessage format.

    Also supports converting parts of a message (with limitations then of course)

    Args:
        message: Message to convert (ModelMessage or its parts)
        tools: Original ToolInfo set to enrich ToolCallInfos with additional info
        agent_name: Name of the agent of this message

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
                    case SystemPromptPart():
                        content_parts.append(str(part.content))
            return ChatMessage(content="\n".join(content_parts), role=role)

        case ModelResponse():
            # Collect content and tool calls from all parts
            tool_calls = get_tool_calls([message], tools, None)
            parts = [format_part(p) for p in message.parts if isinstance(p, TextPart)]
            content = "\n".join(parts)
            return ChatMessage(content=content, role="assistant", tool_calls=tool_calls)

        case TextPart() | UserPromptPart() | SystemPromptPart() as part:
            role = "assistant" if isinstance(part, TextPart) else "user"
            return ChatMessage(content=format_part(part), role=role)

        case ToolCallPart():
            args = (
                message.args.args_dict
                if isinstance(message.args, ArgsDict)
                else json.loads(message.args.args_json)
            )
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
                else "\n".join(
                    f"- {error['loc']}: {error['msg']}" for error in message.content
                )
            )
            return ChatMessage(content=f"Retry needed: {error_content}", role="assistant")

        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise ValueError(msg)


def to_model_message(message: ChatMessage[str | Content]) -> ModelMessage:
    """Convert ChatMessage to pydantic-ai ModelMessage."""
    match message.content:
        case BaseContent():
            content = [message.content.to_openai_format()]
            return ModelRequest(
                parts=[UserPromptPart(content=json.dumps({"content": content}))]
            )
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
