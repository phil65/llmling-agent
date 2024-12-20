"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from datetime import datetime
import json
from typing import TYPE_CHECKING, Any

from pydantic_ai import messages as _messages
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
import tokonomics

from llmling_agent.log import get_logger
from llmling_agent.models.messages import (
    ChatMessage,
    MessageMetadata,
    TokenAndCostResult,
    TokenUsage,
)


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.result import Usage


async def extract_token_usage_and_cost(
    usage: Usage,
    model: str,
    prompt: str,
    completion: str,
) -> TokenAndCostResult | None:
    """Extract token usage and calculate actual USD cost.

    Args:
        usage: Token counts from pydantic-ai Usage object
        model: Name of the model used
        prompt: The prompt text sent to model
        completion: The completion text received

    Returns:
        Token usage and USD cost, or None if counts unavailable
    """
    if not (
        usage
        and usage.total_tokens is not None
        and usage.request_tokens is not None
        and usage.response_tokens is not None
    ):
        logger.debug("Missing token counts in Usage object")
        return None

    token_usage = TokenUsage(
        total=usage.total_tokens,
        prompt=usage.request_tokens,
        completion=usage.response_tokens,
    )
    logger.debug("Token usage: %s", token_usage)

    cost = await tokonomics.calculate_token_cost(model, token_usage)
    if cost is not None:
        logger.debug("Calculated cost: $%.6f", cost)
        return TokenAndCostResult(token_usage=token_usage, cost_usd=cost)

    logger.debug("Failed to calculate USD cost")
    return None


def format_response(response: str | _messages.ModelRequestPart) -> str:  # noqa: PLR0911
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


def find_last_assistant_message(messages: Sequence[ModelMessage]) -> str | None:
    """Find the last assistant message in history."""
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                match part:
                    case TextPart():
                        return part.content
                    case ToolCallPart() as tool_call:
                        # Format tool calls nicely
                        args = (
                            tool_call.args.args_dict
                            if isinstance(tool_call.args, ArgsDict)
                            else json.loads(tool_call.args.args_json)
                        )
                        return f"Tool: {tool_call.tool_name}\nArgs: {args}"
    return None


def convert_model_message(message: ModelMessage | Any) -> ChatMessage:  # noqa: PLR0911
    """Convert a pydantic-ai message to our ChatMessage format.

    Args:
        message: Message to convert (ModelMessage or its parts)

    Returns:
        Converted ChatMessage

    Raises:
        ValueError: If message type is not supported
    """
    match message:
        case ModelRequest():
            # Use first part's content
            part = message.parts[0]
            return ChatMessage(
                content=str(part.content),
                role="user" if isinstance(part, UserPromptPart) else "system",
                timestamp=datetime.now(),
            )

        case ModelResponse():
            # Convert first part (shouldn't have multiple typically)
            return convert_model_message(message.parts[0])

        case TextPart():
            return ChatMessage(
                content=message.content,
                role="assistant",
                timestamp=datetime.now(),
            )

        case ToolCallPart():
            args = (
                message.args.args_dict
                if isinstance(message.args, ArgsDict)
                else message.args.args_json
            )
            return ChatMessage(
                content=f"Tool call: {message.tool_name}\nArgs: {args}",
                role="assistant",
                metadata=MessageMetadata(tool=message.tool_name),
                timestamp=datetime.now(),
            )

        case ToolReturnPart():
            return ChatMessage(
                content=f"Tool {message.tool_name} returned: {message.content}",
                role="assistant",
                metadata=MessageMetadata(tool=message.tool_name),
                timestamp=datetime.now(),
            )

        case RetryPromptPart():
            error_content = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(
                    f"- {error['loc']}: {error['msg']}" for error in message.content
                )
            )
            return ChatMessage(
                content=f"Retry needed: {error_content}",
                role="assistant",
                timestamp=datetime.now(),
            )

        case SystemPromptPart():
            return ChatMessage(
                content=message.content,
                role="system",
                timestamp=datetime.now(),
            )

        case UserPromptPart():
            return ChatMessage(
                content=message.content,
                role="user",
                timestamp=datetime.now(),
            )

        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise ValueError(msg)
