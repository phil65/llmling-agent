"""Helper functions for common message processing logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentpool.messaging import ChatMessage
from agentpool.prompts.convert import convert_prompts


if TYPE_CHECKING:
    from pydantic_ai import UserContent

    from agentpool.common_types import PromptCompatible
    from agentpool.messaging import MessageNode
    from agentpool.messaging.connection_manager import ConnectionManager


async def prepare_prompts(
    *prompt: PromptCompatible,
    parent_id: str | None = None,
    session_id: str | None = None,
) -> tuple[ChatMessage[Any], list[UserContent]]:
    """Prepare prompts for processing.

    Args:
        *prompt: The prompt(s) to prepare.
        parent_id: Optional ID of the parent message (typically the previous response).
        session_id: Optional conversation ID for the user message.

    Returns:
        A tuple of:
            - A ChatMessage representing the user prompt.
            - A list of prompts to be sent to the model.
    """
    prompts = await convert_prompts(prompt)
    user_msg = ChatMessage.user_prompt(message=prompts, parent_id=parent_id, session_id=session_id)
    return user_msg, prompts


async def finalize_message(
    message: ChatMessage[Any],
    previous_message: ChatMessage[Any] | None,
    node: MessageNode[Any, Any],
    connections: ConnectionManager,
    wait_for_connections: bool | None = None,
) -> ChatMessage[Any]:
    """Handle message finalization and routing.

    Args:
        message: The response message to finalize
        previous_message: The original user message (if any)
        node: The message node that produced the message
        connections: Connection manager for routing
        wait_for_connections: Whether to wait for connected nodes

    Returns:
        The finalized message
    """
    await node.message_sent.emit(message)  # Emit signals
    await node.log_message(message)  # Log message if enabled
    # Route to connections
    await connections.route_message(message, wait=wait_for_connections)
    return message
