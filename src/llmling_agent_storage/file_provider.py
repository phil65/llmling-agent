"""File provider implementation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypedDict, cast

from tokonomics.toko_types import TokenUsage
from upath import UPath

from llmling_agent.common_types import JsonValue, MessageRole  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage, TokenCost
from llmling_agent.utils.now import get_now
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from llmling_agent.tools import ToolCallInfo
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


class ConversationData(TypedDict):
    """Data structure for storing conversation information."""

    id: str
    agent_name: str
    start_time: str


class ToolCallData(TypedDict):
    """Data structure for storing tool call information."""

    conversation_id: str
    message_id: str
    tool_name: str
    args: dict[str, Any]
    result: str
    timestamp: str


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
    tool_calls: list[ToolCallData]
    commands: list[CommandData]


class FileProvider(StorageProvider):
    """File-based storage using various formats.

    Automatically detects format from file extension or uses specified format.
    Supported formats: YAML (.yml, .yaml), JSON (.json), TOML (.toml)
    """

    can_load_history = True

    def __init__(self, config: FileStorageConfig):
        """Initialize file provider.

        Args:
            config: Configuration for provider
        """
        super().__init__(config)
        self.path = UPath(config.path)
        self.format = config.format
        self.encoding = config.encoding
        self._data: StorageData = {
            "messages": [],
            "conversations": [],
            "tool_calls": [],
            "commands": [],
        }
        self._load()

    def _load(self):
        """Load data from file if it exists."""
        import yamling

        if self.path.exists():
            self._data = yamling.load_file(
                self.path,
                self.format,  # pyright: ignore
                verify_type=StorageData,
            )
        self._save()

    def _save(self):
        """Save current data to file."""
        import yamling

        self.path.parent.mkdir(parents=True, exist_ok=True)
        yamling.dump_file(self._data, self.path, mode=self.format)  # pyright: ignore

    def cleanup(self):
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
            if query.until and datetime.fromisoformat(
                msg["timestamp"]
            ) > datetime.fromisoformat(query.until):
                continue
            if query.contains and query.contains not in msg["content"]:
                continue
            if query.roles and msg["role"] not in query.roles:
                continue

            # Convert to ChatMessage
            cost_info = None
            if msg["token_usage"]:
                usage = cast(TokenUsage, msg["token_usage"])
                cost = Decimal(msg["cost"] or 0.0)
                cost_info = TokenCost(token_usage=usage, total_cost=cost)

            chat_message = ChatMessage[str](
                content=msg["content"],
                conversation_id=msg["conversation_id"],
                role=cast(MessageRole, msg["role"]),
                name=msg["name"],
                model=msg["model"],
                cost_info=cost_info,
                response_time=msg["response_time"],
                forwarded_from=msg["forwarded_from"] or [],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
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
    ):
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
            "token_usage": cost_info.token_usage.copy() if cost_info else None,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
        })
        self._save()

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ):
        """Log a new conversation."""
        conversation: ConversationData = {
            "id": conversation_id,
            "agent_name": node_name,
            "start_time": (start_time or get_now()).isoformat(),
        }
        self._data["conversations"].append(conversation)
        self._save()

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log a tool call."""
        call: ToolCallData = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "tool_name": tool_call.tool_name,
            "args": tool_call.args,
            "result": str(tool_call.result),
            "timestamp": tool_call.timestamp.isoformat(),
        }
        self._data["tool_calls"].append(call)
        self._save()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ):
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
                "tool_calls": [],
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
                    c["id"]
                    for c in self._data["conversations"]
                    if c["agent_name"] == agent_name
                }
            ]
        else:
            # Clear all
            self._data["messages"].clear()
            self._data["conversations"].clear()
            self._data["tool_calls"].clear()
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
                in {
                    c["id"]
                    for c in self._data["conversations"]
                    if c["agent_name"] == agent_name
                }
            )
        else:
            conv_count = len(self._data["conversations"])
            msg_count = len(self._data["messages"])

        return conv_count, msg_count
