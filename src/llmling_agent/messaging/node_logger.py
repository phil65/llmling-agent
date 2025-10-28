"""Logging functionality for node interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from llmling_agent import log


if TYPE_CHECKING:
    from llmling_agent.messaging.messageemitter import MessageEmitter
    from llmling_agent.messaging.messages import ChatMessage


logger = log.get_logger(__name__)


class NodeLogger:
    """Handles database logging for node interactions."""

    def __init__(self, node: MessageEmitter[Any, Any], enable_db_logging: bool = True):
        """Initialize logger.

        Args:
            node: Node to log interactions for
            enable_db_logging: Whether to enable logging
        """
        self.node = node
        self.enable_db_logging = enable_db_logging
        self.conversation_id = str(uuid4())
        # Initialize conversation record if enabled
        if enable_db_logging:
            self.node.context.storage.log_conversation.sync(
                conversation_id=self.conversation_id,
                node_name=self.node.name,
            )

            # Connect to the combined signal to capture all messages
            # TODO: need to check this
            # node.message_received.connect(self.log_message)
            node.message_sent.connect(self.log_message)

    async def get_message_history(self, limit: int | None = None) -> list[ChatMessage]:
        """Get message history from storage."""
        if not self.enable_db_logging:
            return []  # No history if not logging

        from llmling_agent_config.session import SessionQuery

        return await self.node.context.storage.filter_messages(
            SessionQuery(name=self.conversation_id, limit=limit)
        )

    def log_message(self, message: ChatMessage):
        """Handle message from chat signal."""
        if not self.enable_db_logging:
            return
        self.node.context.storage.log_message.sync(
            message_id=message.message_id,
            conversation_id=message.conversation_id or "",
            content=str(message.content),
            role=message.role,
            name=message.name,
            cost_info=message.cost_info,
            model=message.model,
            response_time=message.response_time,
            forwarded_from=message.forwarded_from,
        )
