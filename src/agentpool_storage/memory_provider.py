"""In-memory storage provider for testing."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from agentpool.messaging import ChatMessage, TokenCost
from agentpool.storage import deserialize_messages
from agentpool.utils.time_utils import get_now
from agentpool_storage.base import StorageProvider
from agentpool_storage.models import ConversationData


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentpool.common_types import JsonValue
    from agentpool.sessions.models import ProjectData
    from agentpool_config.session import SessionQuery
    from agentpool_config.storage import MemoryStorageConfig
    from agentpool_storage.models import MessageData, QueryFilters, StatsFilters, TokenUsage


def _dict_to_chat_message(msg: dict[str, Any]) -> ChatMessage[str]:
    """Convert a stored message dict to ChatMessage."""
    cost_info = None
    if msg.get("cost_info"):
        cost_info = TokenCost(token_usage=msg["cost_info"], total_cost=msg.get("cost", 0.0))

    # Build kwargs, only including timestamp/message_id if they exist
    kwargs: dict[str, Any] = {
        "content": msg["content"],
        "role": msg["role"],
        "name": msg.get("name"),
        "model_name": msg.get("model"),
        "cost_info": cost_info,
        "response_time": msg.get("response_time"),
        "parent_id": msg.get("parent_id"),
        "session_id": msg.get("session_id"),
    }
    if msg.get("timestamp"):
        kwargs["timestamp"] = msg["timestamp"]
    if msg.get("message_id"):
        kwargs["message_id"] = msg["message_id"]

    return ChatMessage[str](**kwargs)


class MemoryStorageProvider(StorageProvider):
    """In-memory storage provider for testing."""

    can_load_history = True

    def __init__(self, config: MemoryStorageConfig) -> None:
        super().__init__(config)
        self.messages: list[dict[str, Any]] = []
        self.conversations: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []
        self.projects: dict[str, ProjectData] = {}

    def cleanup(self) -> None:
        """Clear all stored data."""
        self.messages.clear()
        self.conversations.clear()
        self.commands.clear()
        self.projects.clear()

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages from memory."""
        from agentpool.messaging import ChatMessage

        filtered = []
        for msg in self.messages:
            # Skip if conversation ID doesn't match
            if query.name and msg["session_id"] != query.name:
                continue

            # Skip if agent name doesn't match
            if query.agents and msg["name"] not in query.agents:
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
        session_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        provider_name: str | None = None,
        provider_response_id: str | None = None,
        messages: str | None = None,
        finish_reason: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        """Store message in memory."""
        if next((i for i in self.messages if i["message_id"] == message_id), None):
            msg = f"Duplicate message ID: {message_id}"
            raise ValueError(msg)

        self.messages.append({
            "session_id": session_id,
            "message_id": message_id,
            "parent_id": parent_id,
            "content": content,
            "role": role,
            "name": name,
            "cost_info": cost_info.token_usage if cost_info else None,
            "model": model,
            "response_time": response_time,
            "provider_name": provider_name,
            "provider_response_id": provider_response_id,
            "messages": messages,
            "finish_reason": finish_reason,
            "timestamp": get_now(),
        })

    async def log_session(
        self,
        *,
        session_id: str,
        node_name: str,
        start_time: datetime | None = None,
        model: str | None = None,
    ) -> None:
        """Store conversation in memory."""
        if next((i for i in self.conversations if i["id"] == session_id), None):
            msg = f"Duplicate conversation ID: {session_id}"
            raise ValueError(msg)
        self.conversations.append({
            "id": session_id,
            "agent_name": node_name,
            "title": None,
            "start_time": start_time or get_now(),
        })

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a conversation."""
        for conv in self.conversations:
            if conv["id"] == session_id:
                conv["title"] = title
                return

    async def get_session_title(self, session_id: str) -> str | None:
        """Get the title of a conversation."""
        for conv in self.conversations:
            if conv["id"] == session_id:
                return conv.get("title")
        return None

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a session."""
        messages: list[ChatMessage[str]] = []
        for msg in self.messages:
            if msg.get("session_id") != session_id:
                continue
            messages.append(_dict_to_chat_message(msg))

        # Sort by timestamp
        now = get_now()
        messages.sort(key=lambda m: m.timestamp or now)

        if not include_ancestors or not messages:
            return messages

        # Get ancestor chain if first message has parent_id
        first_msg = messages[0]
        if first_msg.parent_id:
            ancestors = await self.get_message_ancestry(first_msg.parent_id, session_id=session_id)
            return ancestors + messages

        return messages

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[str] | None:
        """Get a single message by ID."""
        for msg in self.messages:
            if msg.get("message_id") == message_id:
                return _dict_to_chat_message(msg)
        return None

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message."""
        ancestors: list[ChatMessage[str]] = []
        current_id: str | None = message_id

        while current_id:
            msg = await self.get_message(current_id, session_id=session_id)
            if not msg:
                break
            ancestors.append(msg)
            current_id = msg.parent_id

        # Reverse to get oldest first
        ancestors.reverse()
        return ancestors

    async def fork_conversation(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        """Fork a conversation at a specific point."""
        # Find source conversation
        source_conv = next((c for c in self.conversations if c["id"] == source_session_id), None)
        if not source_conv:
            msg = f"Source conversation not found: {source_session_id}"
            raise ValueError(msg)

        # Determine fork point
        fork_point_id: str | None = None
        if fork_from_message_id:
            # Verify message exists in source conversation
            msg_exists = any(
                m.get("message_id") == fork_from_message_id and m["session_id"] == source_session_id
                for m in self.messages
            )
            if not msg_exists:
                err = f"Message {fork_from_message_id} not found in conversation"
                raise ValueError(err)
            fork_point_id = fork_from_message_id
        else:
            # Find last message in source conversation
            conv_messages = [m for m in self.messages if m["session_id"] == source_session_id]
            if conv_messages:
                now = get_now()
                conv_messages.sort(key=lambda m: m.get("timestamp") or now)
                fork_point_id = conv_messages[-1].get("message_id")

        # Create new conversation
        agent_name = new_agent_name or source_conv["agent_name"]
        title = (
            f"{source_conv.get('title') or 'Conversation'} (fork)"
            if source_conv.get("title")
            else None
        )
        self.conversations.append({
            "id": new_session_id,
            "agent_name": agent_name,
            "title": title,
            "start_time": get_now(),
        })

        return fork_point_id

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

    async def get_sessions(
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
                if msg["session_id"] != conv_id:
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

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
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

    async def reset(self, *, agent_name: str | None = None, hard: bool = False) -> tuple[int, int]:
        """Reset stored data."""
        # Get counts first
        conv_count, msg_count = await self.get_session_counts(agent_name=agent_name)

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
                if m["session_id"]
                not in {c["id"] for c in self.conversations if c["agent_name"] == agent_name}
            ]
        else:
            # Clear all
            self.messages.clear()
            self.conversations.clear()
            self.commands.clear()

        return conv_count, msg_count

    async def get_session_counts(self, *, agent_name: str | None = None) -> tuple[int, int]:
        """Get conversation and message counts."""
        if agent_name:
            conv_count = sum(1 for c in self.conversations if c["agent_name"] == agent_name)
            msg_count = sum(
                1
                for m in self.messages
                if any(
                    c["id"] == m["session_id"] and c["agent_name"] == agent_name
                    for c in self.conversations
                )
            )
        else:
            conv_count = len(self.conversations)
            msg_count = len(self.messages)

        return conv_count, msg_count

    async def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session."""
        original_count = len(self.messages)
        self.messages = [m for m in self.messages if m["session_id"] != session_id]
        return original_count - len(self.messages)

    # Project methods

    async def save_project(self, project: ProjectData) -> None:
        """Save or update a project."""
        self.projects[project.project_id] = project

    async def get_project(self, project_id: str) -> ProjectData | None:
        """Get a project by ID."""
        return self.projects.get(project_id)

    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        """Get a project by worktree path."""
        for project in self.projects.values():
            if project.worktree == worktree:
                return project
        return None

    async def get_project_by_name(self, name: str) -> ProjectData | None:
        """Get a project by friendly name."""
        for project in self.projects.values():
            if project.name == name:
                return project
        return None

    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        """List all projects, ordered by last_active descending."""
        projects = sorted(
            self.projects.values(),
            key=lambda p: p.last_active,
            reverse=True,
        )
        if limit is not None:
            projects = projects[:limit]
        return list(projects)

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        if project_id in self.projects:
            del self.projects[project_id]
            return True
        return False

    async def touch_project(self, project_id: str) -> None:
        """Update project's last_active timestamp."""
        if project_id in self.projects:
            project = self.projects[project_id]
            self.projects[project_id] = project.touch()
