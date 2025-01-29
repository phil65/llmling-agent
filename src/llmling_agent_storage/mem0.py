"""Mem0 storage provider for LLMling agent."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from mem0 import AsyncMemoryClient

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.session import SessionQuery
    from llmling_agent.models.storage import Mem0Config

logger = get_logger(__name__)


class Mem0StorageProvider(StorageProvider):
    """Storage provider using mem0 for conversation history."""

    can_load_history: bool = True
    write_only: bool = False

    def __init__(self, config: Mem0Config):
        """Initialize mem0 client."""
        super().__init__(config)
        self.config: Mem0Config = config
        self.client = AsyncMemoryClient(api_key=config.api_key)

    def _to_mem0_message(
        self,
        content: str,
        role: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert to mem0 message format."""
        return {
            "role": role,
            "content": content,
            "metadata": {
                "name": name,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            },
        }

    def _from_mem0_message(self, msg: dict[str, Any]) -> ChatMessage[str]:
        """Convert from mem0 message format."""
        metadata = msg.get("metadata", {})
        return ChatMessage(
            content=msg["content"],
            role=msg["role"],
            name=metadata.get("name"),
            timestamp=datetime.fromisoformat(
                metadata.get("timestamp", datetime.now().isoformat())
            ),
            metadata=metadata,
        )

    async def log_message(
        self,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: Any | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ) -> None:
        """Log a message to mem0."""
        metadata = {
            "model": model,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
        }
        if cost_info:
            metadata.update({
                "token_usage": cost_info.token_usage,
                "total_cost": float(cost_info.total_cost),
            })

        message = self._to_mem0_message(
            content=content,
            role=role,
            name=name,
            metadata=metadata,
        )
        await self.client.add(
            [message],
            user_id=conversation_id,
            output_format=self.config.output_format,
        )

    async def log_conversation(
        self,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log conversation metadata."""
        metadata = {
            "type": "conversation_start",
            "agent_name": agent_name,
            "start_time": (start_time or datetime.now()).isoformat(),
        }
        message = self._to_mem0_message(
            content="Conversation started",
            role="system",
            metadata=metadata,
        )
        await self.client.add(
            [message],
            user_id=conversation_id,
            output_format=self.config.output_format,
        )

    async def log_tool_call(
        self,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ) -> None:
        """Log tool usage."""
        metadata = {
            "type": "tool_call",
            "tool_name": tool_call.tool_name,
            "args": tool_call.args,
            "result": tool_call.result,
            "error": tool_call.error,
            "timing": tool_call.timing,
            "message_id": message_id,
        }
        message = self._to_mem0_message(
            content=f"Tool call: {tool_call.tool_name}",
            role="system",
            metadata=metadata,
        )
        await self.client.add(
            [message],
            user_id=conversation_id,
            output_format=self.config.output_format,
        )

    async def filter_messages(
        self,
        query: SessionQuery,
    ) -> list[ChatMessage[str]]:
        """Search conversation history."""
        if not query.name:
            return []  # mem0 requires a user_id

        filters: dict[str, Any] = {"AND": [{"user_id": query.name}]}

        # Add time filters if specified
        # Note: query.since/until are datetime objects here
        if query.since or query.until:
            date_filter = {}
            if query.since:
                date_filter["gte"] = datetime.fromisoformat(str(query.since)).isoformat()
            if query.until:
                date_filter["lte"] = datetime.fromisoformat(str(query.until)).isoformat()
            filters["AND"].append({"created_at": date_filter})

        # Add role filters
        if query.roles:
            filters["AND"].append({"role": {"in": list(query.roles)}})

        # Use v2 search with filters
        results = await self.client.search(
            query=query.contains or "",  # Search text
            version="v2",
            filters=filters,
            output_format=self.config.output_format,
        )

        # Convert to ChatMessage format
        messages = [self._from_mem0_message(msg) for msg in results]

        # Apply limit if specified
        if query.limit:
            messages = messages[: query.limit]

        return messages

    async def cleanup(self) -> None:
        """Clean up by resetting client."""
        await self.client.reset()
