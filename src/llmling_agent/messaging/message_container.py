"""Message container with statistics and formatting capabilities."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Literal

from psygnal.containers import EventedList

from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage
from llmling_agent.utils.count_tokens import batch_count_tokens, count_tokens


if TYPE_CHECKING:
    from datetime import datetime

    from llmling_agent.common_types import MessageRole
    from llmling_agent.messaging.messages import FormatStyle
    from llmling_agent.utils.dag import DAGNode


logger = get_logger(__name__)
DEFAULT_TOKEN_MODEL = "gpt-4.1"


class ChatMessageList(EventedList[ChatMessage[Any]]):
    """Container for tracking and managing chat messages.

    Extends EventedList to provide:
    - Message statistics (tokens, costs)
    - History formatting
    - Token-aware context window management
    - Role-based filtering
    """

    def get_message_tokens(self, message: ChatMessage[Any]) -> int:
        """Get token count for a single message.

        Uses cost_info if available, falls back to tiktoken estimation.

        Args:
            message: Message to count tokens for

        Returns:
            Token count for the message
        """
        if message.cost_info:
            return message.cost_info.token_usage.total_tokens
        return count_tokens(str(message.usage.total_tokens), message.model_name)

    def get_history_tokens(self, fallback_model: str | None = None) -> int:
        """Get total token count for all messages.

        Uses cost_info when available, falls back to tiktoken estimation
        for messages without usage information.

        Returns:
            Total token count across all messages
        """
        # Use cost_info if available
        total = sum(m.cost_info.token_usage.total_tokens for m in self if m.cost_info)
        # For messages without cost_info, estimate using tiktoken
        if msgs := [msg for msg in self if not msg.cost_info]:
            if fallback_model:
                model_name = fallback_model
            else:
                model_name = next((m.model_name for m in self if m.model_name), DEFAULT_TOKEN_MODEL)
            contents = [str(msg.content) for msg in msgs]
            total += sum(batch_count_tokens(contents, model_name))

        return total

    def get_total_cost(self) -> float:
        """Calculate total cost in USD across all messages.

        Only includes messages with cost information.

        Returns:
            Total cost in USD
        """
        return sum(float(msg.cost_info.total_cost) for msg in self if msg.cost_info)

    @property
    def last_message(self) -> ChatMessage[Any] | None:
        """Get most recent message or None if empty."""
        return self[-1] if self else None

    def format(self, *, style: FormatStyle = "simple", **kwargs: Any) -> str:
        """Format conversation history with configurable style.

        Args:
            style: Formatting style to use
            **kwargs: Additional formatting options passed to message.format()

        Returns:
            Formatted conversation history as string
        """
        return "\n".join(msg.format(style=style, **kwargs) for msg in self)

    def filter_by_role(
        self,
        role: MessageRole,
        *,
        max_messages: int | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get messages with specific role.

        Args:
            role: Role to filter by (user/assistant/system)
            max_messages: Optional limit on number of messages to return

        Returns:
            List of messages with matching role
        """
        messages = [msg for msg in self if msg.role == role]
        return messages[-max_messages:] if max_messages else messages

    def get_between(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get messages within a time range.

        Args:
            start_time: Optional start of range
            end_time: Optional end of range

        Returns:
            List of messages within the time range
        """
        messages = list(self)
        if start_time:
            messages = [msg for msg in messages if msg.timestamp >= start_time]
        if end_time:
            messages = [msg for msg in messages if msg.timestamp <= end_time]
        return messages

    def _build_flow_dag(self, message: ChatMessage[Any]) -> DAGNode | None:
        """Build DAG from message flow.

        Args:
            message: Message to build flow DAG for

        Returns:
            Root DAGNode of the graph
        """
        from llmling_agent.utils.dag import DAGNode

        # Get messages from this conversation
        conv_messages = [m for m in self if m.conversation_id == message.conversation_id]
        nodes: dict[str, DAGNode] = {}
        for msg in conv_messages:  # First create all nodes
            if msg.forwarded_from:
                chain = [*msg.forwarded_from, msg.name or "unknown"]
                for name in chain:
                    if name not in nodes:
                        nodes[name] = DAGNode(name)

        # Then set up parent relationships
        for msg in conv_messages:
            if msg.forwarded_from:
                chain = [*msg.forwarded_from, msg.name or "unknown"]
                # Connect consecutive nodes
                for parent_name, child_name in itertools.pairwise(chain):
                    parent = nodes[parent_name]
                    child = nodes[child_name]
                    if parent not in child.parents:
                        child.add_parent(parent)

        # Find root nodes (those without parents)
        roots = [node for node in nodes.values() if node.is_root]
        if not roots:
            return None
        return roots[0]  # Return first root for now

    def to_mermaid_graph(
        self,
        message: ChatMessage[Any],
        *,
        title: str = "",
        theme: str | None = None,
        rankdir: Literal["TB", "BT", "LR", "RL"] = "LR",
    ) -> str:
        """Convert message flow to mermaid graph."""
        from llmling_agent.utils.dag import dag_to_list

        dag = self._build_flow_dag(message)
        if not dag:
            return ""

        connections = dag_to_list(dag)

        # Convert to mermaid
        lines = ["```mermaid"]
        if title:
            lines.extend(["---", f"title: {title}", "---"])
        if theme:
            lines.append(f'%%{{ init: {{ "theme": "{theme}" }} }}%%')
        lines.append(f"flowchart {rankdir}")

        # Add connections
        for parent, child in connections:
            lines.append(f"    {parent}-->{child}")

        lines.append("```")
        return "\n".join(lines)
