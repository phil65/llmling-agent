"""In-memory storage provider for testing."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from llmling_agent.messaging import ChatMessage, TokenCost
from llmling_agent.storage import deserialize_messages
from llmling_agent.utils.now import get_now
from llmling_agent_storage.base import StorageProvider
from llmling_agent_storage.models import ConversationData


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.common_types import JsonValue
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import MemoryStorageConfig
    from llmling_agent_storage.models import MessageData, QueryFilters, StatsFilters, TokenUsage


class MemoryStorageProvider(StorageProvider):
    """In-memory storage provider for testing."""

    can_load_history = True

    def __init__(self, config: MemoryStorageConfig) -> None:
        super().__init__(config)
        self.messages: list[dict[str, Any]] = []
        self.conversations: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []

    def cleanup(self) -> None:
        """Clear all stored data."""
        self.messages.clear()
        self.conversations.clear()
        self.commands.clear()

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages from memory."""
        from llmling_agent.messaging import ChatMessage

        filtered = []
        for msg in self.messages:
            # Skip if conversation ID doesn't match
            if query.name and msg["conversation_id"] != query.name:
                continue

            # Skip if agent name doesn't match
            if query.agents and not (
                msg["name"] in query.agents
                or (
                    query.include_forwarded
                    and msg["forwarded_from"]
                    and any(a in query.agents for a in msg["forwarded_from"])
                )
            ):
                continue

            # Skip if before cutoff time
            if query.since and (cutoff := query.get_time_cutoff()):  # noqa: SIM102
                if msg["timestamp"] < cutoff:
                    continue

            # Skip if after until time
            if query.until and msg["timestamp"] > datetime.fromisoformat(query.until):
                continue

            # Skip if content doesn't match search
            if query.contains and query.contains not in msg["content"]:
                continue

            # Skip if role doesn't match
            if query.roles and msg["role"] not in query.roles:
                continue

            # Convert cost info
            cost_info = None
            if msg["cost_info"]:
                total = msg.get("cost", 0.0)
                cost_info = TokenCost(token_usage=msg["cost_info"], total_cost=total)

            # Create ChatMessage
            chat_message = ChatMessage(
                content=msg["content"],
                role=msg["role"],
                name=msg["name"],
                model_name=msg["model"],
                cost_info=cost_info,
                response_time=msg["response_time"],
                forwarded_from=msg["forwarded_from"] or [],
                timestamp=msg["timestamp"],
                provider_name=msg["provider_name"],
                provider_response_id=msg["provider_response_id"],
                messages=deserialize_messages(msg["messages"]),
                finish_reason=msg["finish_reason"],
            )
            filtered.append(chat_message)

            # Apply limit if specified
            if query.limit and len(filtered) >= query.limit:
                break

        return filtered

    async def log_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
        provider_name: str | None = None,
        provider_response_id: str | None = None,
        messages: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Store message in memory."""
        if next((i for i in self.messages if i["message_id"] == message_id), None):
            msg = f"Duplicate message ID: {message_id}"
            raise ValueError(msg)

        self.messages.append({
            "conversation_id": conversation_id,
            "message_id": message_id,
            "content": content,
            "role": role,
            "name": name,
            "cost_info": cost_info.token_usage if cost_info else None,
            "model": model,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
            "provider_name": provider_name,
            "provider_response_id": provider_response_id,
            "messages": messages,
            "finish_reason": finish_reason,
            "timestamp": get_now(),
        })

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Store conversation in memory."""
        if next((i for i in self.conversations if i["id"] == conversation_id), None):
            msg = f"Duplicate conversation ID: {conversation_id}"
            raise ValueError(msg)
        self.conversations.append({
            "id": conversation_id,
            "agent_name": node_name,
            "title": None,
            "start_time": start_time or get_now(),
        })

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        """Update the title of a conversation."""
        for conv in self.conversations:
            if conv["id"] == conversation_id:
                conv["title"] = title
                return

    async def get_conversation_title(
        self,
        conversation_id: str,
    ) -> str | None:
        """Get the title of a conversation."""
        for conv in self.conversations:
            if conv["id"] == conversation_id:
                return conv.get("title")
        return None

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Store command in memory."""
        self.commands.append({
            "agent_name": agent_name,
            "session_id": session_id,
            "command": command,
            "timestamp": get_now(),
            "context_type": context_type.__name__ if context_type else None,
            "metadata": metadata or {},
        })

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get commands from memory."""
        filtered = []
        for cmd in reversed(self.commands):  # newest first
            if current_session_only and cmd["session_id"] != session_id:
                continue
            if not current_session_only and cmd["agent_name"] != agent_name:
                continue
            filtered.append(cmd["command"])
            if limit and len(filtered) >= limit:
                break
        return filtered

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations from memory."""
        results: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []

        # First get matching conversations
        convs = {}
        for conv in self.conversations:
            if filters.agent_name and conv["agent_name"] != filters.agent_name:
                continue
            if filters.since and conv["start_time"] < filters.since:
                continue
            convs[conv["id"]] = conv

        # Then get messages for each conversation
        for conv_id, conv in convs.items():
            conv_messages: list[ChatMessage[str]] = []
            for msg in self.messages:
                if msg["conversation_id"] != conv_id:
                    continue
                if filters.query and filters.query not in msg["content"]:
                    continue
                if filters.model and msg["model"] != filters.model:
                    continue

                cost_info = None
                if msg["cost_info"]:
                    total = msg.get("cost", 0.0)
                    cost_info = TokenCost(token_usage=msg["cost_info"], total_cost=total)

                chat_msg = ChatMessage[str](
                    content=msg["content"],
                    role=msg["role"],
                    name=msg["name"],
                    model_name=msg["model"],
                    cost_info=cost_info,
                    response_time=msg["response_time"],
                    forwarded_from=msg["forwarded_from"],
                    timestamp=msg["timestamp"],
                )
                conv_messages.append(chat_msg)

            # Skip if no matching messages for content filter
            if filters.query and not conv_messages:
                continue

            # Convert ChatMessages to MessageData format for ConversationData
            message_data: list[MessageData] = [
                cast(
                    "MessageData",
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "model": msg.model_name,
                        "name": msg.name,
                        "token_usage": msg.cost_info.token_usage if msg.cost_info else None,
                        "cost": msg.cost_info.total_cost if msg.cost_info else None,
                        "response_time": msg.response_time,
                    },
                )
                for msg in conv_messages
            ]

            # Create conversation data with proper MessageData
            conv_data = ConversationData(
                id=conv_id,
                agent=conv["agent_name"],
                title=conv.get("title"),
                start_time=conv["start_time"].isoformat(),
                messages=message_data,  # Now using properly typed MessageData
                token_usage=self._aggregate_token_usage(conv_messages),
            )
            results.append((conv_data, conv_messages))

            if filters.limit and len(results) >= filters.limit:
                break

        return results

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get statistics from memory."""
        # Collect raw data
        rows = []
        for msg in self.messages:
            if msg["timestamp"] <= filters.cutoff:
                continue
            if filters.agent_name and msg["name"] != filters.agent_name:
                continue

            cost_info = None
            if msg["cost_info"]:
                total = msg.get("cost", 0.0)
                cost_info = TokenCost(token_usage=msg["cost_info"], total_cost=total)

            rows.append((msg["model"], msg["name"], msg["timestamp"], cost_info))

        # Use base class aggregation
        return self.aggregate_stats(rows, filters.group_by)

    @staticmethod
    def _aggregate_token_usage(messages: Sequence[ChatMessage[Any]]) -> TokenUsage:
        """Sum up tokens from a sequence of messages."""
        total = prompt = completion = 0
        for msg in messages:
            if msg.cost_info:
                total += msg.cost_info.token_usage.total_tokens
                prompt += msg.cost_info.token_usage.input_tokens
                completion += msg.cost_info.token_usage.output_tokens
        return {"total": total, "prompt": prompt, "completion": completion}

    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        """Reset stored data."""
        # Get counts first
        conv_count, msg_count = await self.get_conversation_counts(agent_name=agent_name)

        if hard:
            if agent_name:
                msg = "Hard reset cannot be used with agent_name"
                raise ValueError(msg)
            # Clear everything
            self.cleanup()
            return conv_count, msg_count

        if agent_name:
            # Filter out data for specific agent
            self.conversations = [c for c in self.conversations if c["agent_name"] != agent_name]
            self.messages = [
                m
                for m in self.messages
                if m["conversation_id"]
                not in {c["id"] for c in self.conversations if c["agent_name"] == agent_name}
            ]
        else:
            # Clear all
            self.messages.clear()
            self.conversations.clear()
            self.commands.clear()

        return conv_count, msg_count

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get conversation and message counts."""
        if agent_name:
            conv_count = sum(1 for c in self.conversations if c["agent_name"] == agent_name)
            msg_count = sum(
                1
                for m in self.messages
                if any(
                    c["id"] == m["conversation_id"] and c["agent_name"] == agent_name
                    for c in self.conversations
                )
            )
        else:
            conv_count = len(self.conversations)
            msg_count = len(self.messages)

        return conv_count, msg_count
