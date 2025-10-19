"""Mem0 storage provider for LLMling agent."""

from __future__ import annotations

from datetime import datetime
import os
from typing import TYPE_CHECKING, Any

from mem0 import AsyncMemoryClient

from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.utils.now import get_now
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from llmling_agent.tools import ToolCallInfo
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import Mem0Config

logger = get_logger(__name__)


class Mem0StorageProvider(StorageProvider):
    """Storage provider using mem0 for conversation history."""

    can_load_history: bool = True
    write_only: bool = False

    def __init__(self, config: Mem0Config):
        """Initialize mem0 client."""
        super().__init__(config)
        self.config: Mem0Config = config
        key = (
            self.config.api_key.get_secret_value()
            if self.config.api_key
            else os.getenv("MEM0_API_KEY")
        )
        self.client = AsyncMemoryClient(api_key=key)

    def _to_mem0_message(
        self,
        content: str,
        role: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert to mem0 message format."""
        ts = get_now().isoformat()
        meta = {"name": name, "timestamp": ts, **(metadata or {})}
        return {"role": role, "content": content, "metadata": meta}

    def _from_mem0_message(self, msg: dict[str, Any]) -> ChatMessage[str]:
        """Convert from mem0 message format."""
        metadata = msg.get("metadata", {})
        ts = datetime.fromisoformat(metadata.get("timestamp", get_now().isoformat()))
        return ChatMessage(
            content=msg["content"],
            role=msg["role"],
            name=metadata.get("name"),
            timestamp=ts,
            metadata=metadata,
        )

    async def log_message(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: Any | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ):
        """Log a message to mem0."""
        metadata = {
            "model": model,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
            "message_id": message_id,
        }
        if cost_info:
            total = float(cost_info.total_cost)
            metadata.update({"token_usage": cost_info.token_usage, "total_cost": total})

        message = self._to_mem0_message(
            content=content,
            role=role,
            name=name,
            metadata=metadata,
        )
        fmt = self.config.output_format
        await self.client.add([message], user_id=conversation_id, output_format=fmt)

    async def log_conversation(
        self,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ):
        """Log conversation metadata."""
        t = (start_time or get_now()).isoformat()
        meta = {"type": "conversation_start", "agent_name": node_name, "start_time": t}
        message = self._to_mem0_message("Conversation started", "system", metadata=meta)
        fmt = self.config.output_format
        await self.client.add([message], user_id=conversation_id, output_format=fmt)

    async def log_tool_call(
        self,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
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
        content = f"Tool call: {tool_call.tool_name}"
        message = self._to_mem0_message(content=content, role="system", metadata=metadata)
        fmt = self.config.output_format
        await self.client.add([message], user_id=conversation_id, output_format=fmt)

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
        text = query.contains or ""
        fmt = self.config.output_format
        results = await self.client.search(text, filters=filters, output_format=fmt)
        # Convert to ChatMessage format
        messages = [self._from_mem0_message(msg) for msg in results.values()]

        # Apply limit if specified
        if query.limit:
            messages = messages[: query.limit]

        return messages

    async def cleanup(self):
        """Clean up by resetting client."""
        await self.client.reset()
