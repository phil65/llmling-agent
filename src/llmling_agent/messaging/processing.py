"""Helper functions for common message processing logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.messaging import ChatMessage
from llmling_agent.prompts.convert import convert_prompts


if TYPE_CHECKING:
    from pydantic_ai import UserContent

    from llmling_agent.common_types import PromptCompatible
    from llmling_agent.messaging import MessageNode
    from llmling_agent.messaging.connection_manager import ConnectionManager


async def prepare_prompts(
    *prompt: PromptCompatible | ChatMessage[Any],
) -> tuple[ChatMessage[Any], list[UserContent], ChatMessage[Any] | None]:
    """Prepare prompts for processing.

    Extracted from MessageNode.pre_run logic.

    Args:
        *prompt: The prompt(s) to prepare.

    Returns:
        A tuple of:
            - Either incoming message, or a constructed incoming message based
              on the prompt(s).
            - A list of prompts to be sent to the model.
            - The original ChatMessage if forwarded, None otherwise
    """
    if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
        original_msg = prompt[0]
        # Update received message's chain to show it came through its source
        user_msg = original_msg.forwarded(original_msg).to_request()
        prompts = await convert_prompts([user_msg.content])
        # clear cost info to avoid double-counting
        return user_msg, prompts, original_msg
    prompts = await convert_prompts(prompt)
    user_msg = ChatMessage.user_prompt(message=prompts)
    return user_msg, prompts, None


async def finalize_message(
    message: ChatMessage[Any],
    previous_message: ChatMessage[Any] | None,
    node: MessageNode[Any, Any],
    connections: ConnectionManager,
    original_message: ChatMessage[Any] | None,
    wait_for_connections: bool | None = None,
) -> ChatMessage[Any]:
    """Handle message finalization and routing.

    Args:
        message: The response message to finalize
        previous_message: The original user message (if any)
        node: The message node that produced the message
        connections: Connection manager for routing
        original_message: The original ChatMessage if forwarded, None otherwise
        wait_for_connections: Whether to wait for connected nodes

    Returns:
        The finalized message
    """
    # For chain processing, update the response's chain if input was forwarded
    if original_message:
        message = message.forwarded(original_message)
    node.message_sent.emit(message)  # Emit signals
    await node.log_message(message)  # Log message if enabled
    # Route to connections
    await connections.route_message(message, wait=wait_for_connections)
    return message
