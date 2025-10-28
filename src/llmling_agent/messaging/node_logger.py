"""Logging functionality for node interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from llmling_agent import log
from llmling_agent.messaging.message_container import ChatMessageContainer
from llmling_agent.tools import ToolCallInfo


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
        self.message_history = ChatMessageContainer()
        node.message_received.connect(self.message_history.append)
        node.message_sent.connect(self.message_history.append)
        # Initialize conversation record if enabled
        if enable_db_logging:
            self.init_conversation()
            # Connect to the combined signal to capture all messages
            # TODO: need to check this
            # node.message_received.connect(self.log_message)
            node.message_sent.connect(self.log_message)
            node.tool_used.connect(self.log_tool_call)

    def clear_state(self):
        """Clear node state."""
        self.message_history.clear()

    @property
    def last_message(self) -> ChatMessage[Any] | None:
        """Get last message in history."""
        return self.message_history.last_message

    def init_conversation(self):
        """Create initial conversation record."""
        self.node.context.storage.log_conversation_sync(
            conversation_id=self.conversation_id,
            node_name=self.node.name,
        )

    def log_message(self, message: ChatMessage):
        """Handle message from chat signal."""
        if not self.enable_db_logging:
            return
        self.node.context.storage.log_message_sync(
            message_id=message.message_id,
            conversation_id=message.conversation_id,
            content=str(message.content),
            role=message.role,
            name=message.name,
            cost_info=message.cost_info,
            model=message.model,
            response_time=message.response_time,
            forwarded_from=message.forwarded_from,
        )

    def log_tool_call(self, tool_call: ToolCallInfo):
        """Handle tool usage signal."""
        if not self.enable_db_logging:
            return
        self.node.context.storage.log_tool_call_sync(
            conversation_id=self.conversation_id,
            message_id=tool_call.message_id or "",
            tool_call=tool_call,
        )
