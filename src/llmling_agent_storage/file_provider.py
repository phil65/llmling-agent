"""File provider implementation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, TypedDict, cast

from pydantic_ai import RunUsage
from upathtools import to_upath

from llmling_agent.common_types import JsonValue, MessageRole  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage, TokenCost
from llmling_agent.storage import deserialize_messages
from llmling_agent.utils.now import get_now
from llmling_agent_storage.base import StorageProvider
from llmling_agent_storage.models import TokenUsage


if TYPE_CHECKING:
    from pydantic_ai import FinishReason
    from yamling import FormatType

    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import FileStorageConfig

logger = get_logger(__name__)


class MessageData(TypedDict):
    """Data structure for storing message information."""

    message_id: str
    conversation_id: str
    content: str
    role: str
    timestamp: str
    name: str | None
    model: str | None
    cost: Decimal | None
    token_usage: TokenUsage | None
    response_time: float | None
    forwarded_from: list[str] | None
    provider_name: str | None
    provider_response_id: str | None
    messages: str | None
    finish_reason: FinishReason | None


class ConversationData(TypedDict):
    """Data structure for storing conversation information."""

    id: str
    agent_name: str
    title: str | None
    start_time: str


class CommandData(TypedDict):
    """Data structure for storing command information."""

    agent_name: str
    session_id: str
    command: str
    timestamp: str
    context_type: str | None
    metadata: dict[str, JsonValue]


class StorageData(TypedDict):
    """Data structure for storing storage information."""

    messages: list[MessageData]
    conversations: list[ConversationData]
    commands: list[CommandData]


class FileProvider(StorageProvider):
    """File-based storage using various formats.

    Automatically detects format from file extension or uses specified format.
    Supported formats: YAML (.yml, .yaml), JSON (.json), TOML (.toml)
    """

    can_load_history = True

    def __init__(self, config: FileStorageConfig) -> None:
        """Initialize file provider.

        Args:
            config: Configuration for provider
        """
        super().__init__(config)
        self.path = to_upath(config.path)
        self.format: FormatType = config.format
        self.encoding = config.encoding
        self._data: StorageData = {
            "messages": [],
            "conversations": [],
            "commands": [],
        }
        self._load()

    def _load(self) -> None:
        """Load data from file if it exists."""
        import yamling

        if self.path.exists():
            self._data = yamling.load_file(
                self.path,
                self.format,  # pyright: ignore
                verify_type=StorageData,
            )
        self._save()

    def _save(self) -> None:
        """Save current data to file."""
        import yamling

        self.path.parent.mkdir(parents=True, exist_ok=True)
        yamling.dump_file(self._data, self.path, mode=self.format, overwrite=True)

    def cleanup(self) -> None:
        """Save final state."""
        self._save()

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages = []
        for msg in self._data["messages"]:
            # Apply filters
            if query.name and msg["conversation_id"] != query.name:
                continue
            if query.agents and not (
                msg["name"] in query.agents
                or (
                    query.include_forwarded
                    and msg["forwarded_from"]
                    and any(a in query.agents for a in msg["forwarded_from"])
                )
            ):
                continue
            cutoff = query.get_time_cutoff()
            timestamp = datetime.fromisoformat(msg["timestamp"])
            if query.since and cutoff and (timestamp < cutoff):
                continue
            if query.until and datetime.fromisoformat(msg["timestamp"]) > datetime.fromisoformat(
                query.until
            ):
                continue
            if query.contains and query.contains not in msg["content"]:
                continue
            if query.roles and msg["role"] not in query.roles:
                continue

            # Convert to ChatMessage
            cost_info = None
            if msg["token_usage"]:
                usage = msg["token_usage"]
                cost = Decimal(msg["cost"] or 0.0)
                cost_info = TokenCost(
                    token_usage=RunUsage(
                        input_tokens=usage["prompt"],
                        output_tokens=usage["completion"],
                    ),
                    total_cost=cost,
                )

            chat_message = ChatMessage[str](
                content=msg["content"],
                conversation_id=msg["conversation_id"],
                role=cast(MessageRole, msg["role"]),
                name=msg["name"],
                model_name=msg["model"],
                cost_info=cost_info,
                response_time=msg["response_time"],
                forwarded_from=msg["forwarded_from"] or [],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                provider_name=msg["provider_name"],
                provider_response_id=msg["provider_response_id"],
                messages=deserialize_messages(msg["messages"]),
                finish_reason=msg["finish_reason"],
            )
            messages.append(chat_message)

            if query.limit and len(messages) >= query.limit:
                break

        return messages

    async def log_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
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
        finish_reason: FinishReason | None = None,
    ) -> None:
        """Log a new message."""
        self._data["messages"].append({
            "conversation_id": conversation_id,
            "message_id": message_id,
            "content": content,
            "role": cast(MessageRole, role),
            "timestamp": get_now().isoformat(),
            "name": name,
            "model": model,
            "cost": Decimal(cost_info.total_cost) if cost_info else None,
            "token_usage": TokenUsage(
                prompt=cost_info.token_usage.input_tokens if cost_info else 0,
                completion=cost_info.token_usage.output_tokens if cost_info else 0,
                total=cost_info.token_usage.total_tokens if cost_info else 0,
            ),
            "response_time": response_time,
            "forwarded_from": forwarded_from,
            "provider_name": provider_name,
            "provider_response_id": provider_response_id,
            "messages": messages,
            "finish_reason": finish_reason,
        })
        self._save()

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log a new conversation."""
        conversation: ConversationData = {
            "id": conversation_id,
            "agent_name": node_name,
            "title": None,
            "start_time": (start_time or get_now()).isoformat(),
        }
        self._data["conversations"].append(conversation)
        self._save()

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        """Update the title of a conversation."""
        for conv in self._data["conversations"]:
            if conv["id"] == conversation_id:
                conv["title"] = title
                self._save()
                return

    async def get_conversation_title(
        self,
        conversation_id: str,
    ) -> str | None:
        """Get the title of a conversation."""
        for conv in self._data["conversations"]:
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
        """Log a command execution."""
        cmd: CommandData = {
            "agent_name": agent_name,
            "session_id": session_id,
            "command": command,
            "context_type": context_type.__name__ if context_type else None,
            "metadata": metadata or {},
            "timestamp": get_now().isoformat(),
        }
        self._data["commands"].append(cmd)
        self._save()

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history."""
        commands = []
        for cmd in reversed(self._data["commands"]):  # newest first
            if current_session_only and cmd["session_id"] != session_id:
                continue
            if not current_session_only and cmd["agent_name"] != agent_name:
                continue
            commands.append(cmd["command"])
            if limit and len(commands) >= limit:
                break
        return commands

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
            self._data = {
                "messages": [],
                "conversations": [],
                "commands": [],
            }
            self._save()
            return conv_count, msg_count

        if agent_name:
            # Filter out data for specific agent
            self._data["conversations"] = [
                c for c in self._data["conversations"] if c["agent_name"] != agent_name
            ]
            self._data["messages"] = [
                m
                for m in self._data["messages"]
                if m["conversation_id"]
                not in {
                    c["id"] for c in self._data["conversations"] if c["agent_name"] == agent_name
                }
            ]
        else:
            # Clear all
            self._data["messages"].clear()
            self._data["conversations"].clear()
            self._data["commands"].clear()

        self._save()
        return conv_count, msg_count

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get conversation and message counts."""
        if agent_name:
            conv_count = sum(
                1 for c in self._data["conversations"] if c["agent_name"] == agent_name
            )
            msg_count = sum(
                1
                for m in self._data["messages"]
                if m["conversation_id"]
                in {c["id"] for c in self._data["conversations"] if c["agent_name"] == agent_name}
            )
        else:
            conv_count = len(self._data["conversations"])
            msg_count = len(self._data["messages"])

        return conv_count, msg_count
