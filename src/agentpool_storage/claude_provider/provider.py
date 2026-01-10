"""Claude Code storage provider.

This module implements storage compatible with Claude Code's filesystem format,
enabling interoperability between agentpool and Claude Code.

Key features:
- JSONL-based conversation logs per project
- Multi-agent support (main + sub-agents)
- Message ancestry tracking
- Conversation forking and branching

See ARCHITECTURE.md for detailed documentation of the storage format.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.alias_generators import to_camel
from pydantic_ai import RunUsage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from agentpool.common_types import MessageRole
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage, TokenCost
from agentpool.utils.now import get_now
from agentpool_storage.base import StorageProvider
from agentpool_storage.models import TokenUsage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import FinishReason

    from agentpool_config.session import SessionQuery
    from agentpool_config.storage import ClaudeStorageConfig
    from agentpool_storage.models import ConversationData, MessageData, QueryFilters, StatsFilters

logger = get_logger(__name__)


# Claude JSONL message types

StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
ContentType = Literal["text", "tool_use", "tool_result", "thinking"]
MessageType = Literal[
    "user", "assistant", "queue-operation", "system", "summary", "file-history-snapshot"
]
UserType = Literal["external", "internal"]


class ClaudeBaseModel(BaseModel):
    """Base class for Claude history models."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ClaudeUsage(BaseModel):
    """Token usage from Claude API response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ClaudeMessageContent(BaseModel):
    """Content block in Claude message.

    Supports: text, tool_use, tool_result, thinking blocks.
    """

    type: ContentType
    # For text blocks
    text: str | None = None
    # For tool_use blocks
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    # For tool_result blocks
    tool_use_id: str | None = None
    content: list[dict[str, Any]] | str | None = None  # Can be array or string
    is_error: bool | None = None
    # For thinking blocks
    thinking: str | None = None
    signature: str | None = None


class ClaudeApiMessage(BaseModel):
    """Claude API message structure."""

    model: str
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"]
    content: str | list[ClaudeMessageContent]
    stop_reason: StopReason = None
    usage: ClaudeUsage = Field(default_factory=ClaudeUsage)


class ClaudeUserMessage(BaseModel):
    """User message content."""

    role: Literal["user"]
    content: str | list[ClaudeMessageContent]


class ClaudeMessageEntryBase(ClaudeBaseModel):
    """Base for user/assistant message entries."""

    uuid: str
    parent_uuid: str | None = None
    session_id: str = Field(alias="sessionId")
    timestamp: str
    message: ClaudeApiMessage | ClaudeUserMessage

    # Context (NOT USED directly)
    cwd: str = ""
    git_branch: str = ""
    version: str = ""

    # Metadata (NOT USED)
    user_type: UserType = "external"
    is_sidechain: bool = False
    request_id: str | None = None
    agent_id: str | None = None
    # toolUseResult can be list, dict, or string (error message)
    tool_use_result: list[dict[str, Any]] | dict[str, Any] | str | None = None

    model_config = ConfigDict(populate_by_name=True)


class ClaudeUserEntry(ClaudeMessageEntryBase):
    """User message entry."""

    type: Literal["user"]


class ClaudeAssistantEntry(ClaudeMessageEntryBase):
    """Assistant message entry."""

    type: Literal["assistant"]


class ClaudeQueueOperationEntry(ClaudeBaseModel):
    """Queue operation entry (not a message)."""

    type: Literal["queue-operation"]
    session_id: str
    timestamp: str
    operation: str

    model_config = ConfigDict(populate_by_name=True)


class ClaudeSystemEntry(ClaudeBaseModel):
    """System message entry (context, prompts, etc.)."""

    type: Literal["system"]
    uuid: str
    parent_uuid: str | None = None
    session_id: str
    timestamp: str
    content: str
    subtype: str | None = None
    slug: str | None = None
    level: int | str | None = None
    is_meta: bool = False
    logical_parent_uuid: str | None = None
    compact_metadata: dict[str, Any] | None = None
    # Common fields
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    user_type: UserType = "external"
    is_sidechain: bool = False

    model_config = ConfigDict(populate_by_name=True)


class ClaudeSummaryEntry(ClaudeBaseModel):
    """Summary entry (conversation summary)."""

    type: Literal["summary"]
    leaf_uuid: str
    summary: str

    model_config = ConfigDict(populate_by_name=True)


class ClaudeFileHistoryEntry(ClaudeBaseModel):
    """File history snapshot entry."""

    type: Literal["file-history-snapshot"]
    message_id: str
    snapshot: dict[str, Any]
    is_snapshot_update: bool = False

    model_config = ConfigDict(populate_by_name=True)


# Discriminated union for all entry types
ClaudeJSONLEntry = Annotated[
    ClaudeUserEntry
    | ClaudeAssistantEntry
    | ClaudeQueueOperationEntry
    | ClaudeSystemEntry
    | ClaudeSummaryEntry
    | ClaudeFileHistoryEntry,
    Field(discriminator="type"),
]


class ClaudeStorageProvider(StorageProvider):
    """Storage provider that reads/writes Claude Code's native format.

    Claude stores conversations as JSONL files in:
    - ~/.claude/projects/{path-encoded-project-name}/{session-id}.jsonl

    Each line is a JSON object representing a message in the conversation.

    ## Fields NOT currently used from Claude format:
    - `isSidechain`: Whether message is on a side branch
    - `userType`: Type of user ("external", etc.)
    - `cwd`: Working directory at time of message
    - `gitBranch`: Git branch at time of message
    - `version`: Claude CLI version
    - `requestId`: API request ID
    - `agentId`: Agent identifier for subagents
    - `toolUseResult`: Detailed tool result content (we extract text only)
    - `parentUuid`: Parent message for threading (we use flat history)

    ## Additional Claude data not handled:
    - `~/.claude/todos/`: Todo lists per session
    - `~/.claude/plans/`: Markdown plan files
    - `~/.claude/skills/`: Custom skills
    - `~/.claude/history.jsonl`: Command/prompt history
    """

    can_load_history = True

    def __init__(self, config: ClaudeStorageConfig) -> None:
        """Initialize Claude storage provider.

        Args:
            config: Configuration for the provider
        """
        super().__init__(config)
        self.base_path = Path(config.path).expanduser()
        self.projects_path = self.base_path / "projects"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.projects_path.mkdir(parents=True, exist_ok=True)

    def _encode_project_path(self, path: str) -> str:
        """Encode a project path to Claude's format.

        Claude encodes paths by replacing / with - and prepending -.
        Example: /home/user/project -> -home-user-project
        """
        return path.replace("/", "-")

    def _decode_project_path(self, encoded: str) -> str:
        """Decode a Claude project path back to filesystem path.

        Example: -home-user-project -> /home/user/project
        """
        if encoded.startswith("-"):
            encoded = encoded[1:]
        return "/" + encoded.replace("-", "/")

    def _get_project_dir(self, project_path: str) -> Path:
        """Get the directory for a project's conversations."""
        encoded = self._encode_project_path(project_path)
        return self.projects_path / encoded

    def _list_sessions(self, project_path: str | None = None) -> list[tuple[str, Path]]:
        """List all sessions, optionally filtered by project.

        Returns:
            List of (session_id, file_path) tuples
        """
        sessions = []
        if project_path:
            project_dir = self._get_project_dir(project_path)
            if project_dir.exists():
                for f in project_dir.glob("*.jsonl"):
                    session_id = f.stem
                    sessions.append((session_id, f))
        else:
            for project_dir in self.projects_path.iterdir():
                if project_dir.is_dir():
                    for f in project_dir.glob("*.jsonl"):
                        session_id = f.stem
                        sessions.append((session_id, f))
        return sessions

    def _read_session(self, session_path: Path) -> list[ClaudeJSONLEntry]:
        """Read all entries from a session file."""
        entries: list[ClaudeJSONLEntry] = []
        if not session_path.exists():
            return entries

        adapter = TypeAdapter[Any](ClaudeJSONLEntry)
        with session_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    entry = adapter.validate_python(data)
                    entries.append(entry)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Failed to parse JSONL line", path=str(session_path), error=str(e)
                    )
        return entries

    def _write_entry(self, session_path: Path, entry: ClaudeJSONLEntry) -> None:
        """Append an entry to a session file."""
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with session_path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json(by_alias=True) + "\n")

    def _build_tool_id_mapping(self, entries: list[ClaudeJSONLEntry]) -> dict[str, str]:
        """Build a mapping from tool_call_id to tool_name from assistant entries."""
        mapping: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, ClaudeAssistantEntry):
                continue
            msg = entry.message
            if not isinstance(msg.content, list):
                continue
            for block in msg.content:
                if block.type == "tool_use" and block.id and block.name:
                    mapping[block.id] = block.name
        return mapping

    def _entry_to_chat_message(
        self,
        entry: ClaudeJSONLEntry,
        conversation_id: str,
        tool_id_mapping: dict[str, str] | None = None,
    ) -> ChatMessage[str] | None:
        """Convert a Claude JSONL entry to a ChatMessage.

        Reconstructs pydantic-ai ModelRequest/ModelResponse objects and stores
        them in the messages field for full fidelity.

        Args:
            entry: The JSONL entry to convert
            conversation_id: ID for the conversation
            tool_id_mapping: Optional mapping from tool_call_id to tool_name
                for resolving tool names in ToolReturnPart

        Returns None for non-message entries (queue-operation, summary, etc.).
        """
        # Only handle user/assistant entries with messages
        if not isinstance(entry, (ClaudeUserEntry, ClaudeAssistantEntry)):
            return None

        message = entry.message

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = get_now()

        # Extract display content (text only for UI)
        content = self._extract_text_content(message)

        # Build pydantic-ai message
        pydantic_message = self._build_pydantic_message(
            entry, message, timestamp, tool_id_mapping or {}
        )

        # Extract token usage and cost
        cost_info = None
        model = None
        finish_reason = None
        if isinstance(entry, ClaudeAssistantEntry) and isinstance(message, ClaudeApiMessage):
            usage = message.usage
            input_tokens = (
                usage.input_tokens
                + usage.cache_read_input_tokens
                + usage.cache_creation_input_tokens
            )
            output_tokens = usage.output_tokens

            if input_tokens or output_tokens:
                cost_info = TokenCost(
                    token_usage=RunUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                    total_cost=Decimal(0),  # Claude doesn't store cost directly
                )
            model = message.model
            finish_reason = message.stop_reason

        return ChatMessage[str](
            content=content,
            conversation_id=conversation_id,
            role=entry.type,
            message_id=entry.uuid,
            name="claude" if isinstance(entry, ClaudeAssistantEntry) else None,
            model_name=model,
            cost_info=cost_info,
            timestamp=timestamp,
            parent_id=entry.parent_uuid,
            messages=[pydantic_message] if pydantic_message else [],
            provider_details={"finish_reason": finish_reason} if finish_reason else {},
        )

    def _extract_text_content(self, message: ClaudeApiMessage | ClaudeUserMessage) -> str:
        """Extract text content from a Claude message for display.

        Only extracts text and thinking blocks, not tool calls/results.
        """
        msg_content = message.content
        if isinstance(msg_content, str):
            return msg_content

        text_parts: list[str] = []
        for part in msg_content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            elif part.type == "thinking" and part.thinking:
                # Include thinking in display content
                text_parts.append(f"<thinking>\n{part.thinking}\n</thinking>")
        return "\n".join(text_parts)

    def _build_pydantic_message(
        self,
        entry: ClaudeUserEntry | ClaudeAssistantEntry,
        message: ClaudeApiMessage | ClaudeUserMessage,
        timestamp: datetime,
        tool_id_mapping: dict[str, str],
    ) -> ModelRequest | ModelResponse | None:
        """Build a pydantic-ai ModelRequest or ModelResponse from Claude data.

        Args:
            entry: The entry being converted
            message: The message content
            timestamp: Parsed timestamp
            tool_id_mapping: Mapping from tool_call_id to tool_name
        """
        msg_content = message.content

        if isinstance(entry, ClaudeUserEntry):
            # Build ModelRequest with user prompt parts
            parts: list[UserPromptPart | ToolReturnPart] = []

            if isinstance(msg_content, str):
                parts.append(UserPromptPart(content=msg_content, timestamp=timestamp))
            else:
                for block in msg_content:
                    if block.type == "text" and block.text:
                        parts.append(UserPromptPart(content=block.text, timestamp=timestamp))
                    elif block.type == "tool_result" and block.tool_use_id:
                        # Reconstruct tool return - look up tool name from mapping
                        tool_content = self._extract_tool_result_content(block)
                        tool_name = tool_id_mapping.get(block.tool_use_id, "")
                        parts.append(
                            ToolReturnPart(
                                tool_name=tool_name,
                                content=tool_content,
                                tool_call_id=block.tool_use_id,
                                timestamp=timestamp,
                            )
                        )

            return ModelRequest(parts=parts, timestamp=timestamp) if parts else None

        # Build ModelResponse for assistant
        if not isinstance(message, ClaudeApiMessage):
            return None

        response_parts: list[TextPart | ToolCallPart | ThinkingPart] = []
        usage = RequestUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_read_tokens=message.usage.cache_read_input_tokens,
            cache_write_tokens=message.usage.cache_creation_input_tokens,
        )

        if isinstance(msg_content, str):
            response_parts.append(TextPart(content=msg_content))
        else:
            for block in msg_content:
                if block.type == "text" and block.text:
                    response_parts.append(TextPart(content=block.text))
                elif block.type == "thinking" and block.thinking:
                    response_parts.append(
                        ThinkingPart(
                            content=block.thinking,
                            signature=block.signature,
                        )
                    )
                elif block.type == "tool_use" and block.id and block.name:
                    response_parts.append(
                        ToolCallPart(
                            tool_name=block.name,
                            args=block.input or {},
                            tool_call_id=block.id,
                        )
                    )

        if not response_parts:
            return None

        return ModelResponse(
            parts=response_parts,
            usage=usage,
            model_name=message.model,
            timestamp=timestamp,
        )

    def _extract_tool_result_content(self, block: ClaudeMessageContent) -> str:
        """Extract content from a tool_result block."""
        if block.content is None:
            return ""
        if isinstance(block.content, str):
            return block.content
        # List of content dicts
        text_parts = [
            tc.get("text", "")
            for tc in block.content
            if isinstance(tc, dict) and tc.get("type") == "text"
        ]
        return "\n".join(text_parts)

    def _chat_message_to_entry(
        self,
        message: ChatMessage[str],
        session_id: str,
        parent_uuid: str | None = None,
        cwd: str | None = None,
    ) -> ClaudeUserEntry | ClaudeAssistantEntry:
        """Convert a ChatMessage to a Claude JSONL entry."""
        import uuid

        msg_uuid = message.message_id or str(uuid.uuid4())
        timestamp = (message.timestamp or get_now()).isoformat().replace("+00:00", "Z")

        # Build entry based on role
        if message.role == "user":
            user_msg = ClaudeUserMessage(role="user", content=message.content)
            return ClaudeUserEntry(
                type="user",
                uuid=msg_uuid,
                parent_uuid=parent_uuid,
                sessionId=session_id,
                timestamp=timestamp,
                message=user_msg,
                cwd=cwd or "",
                version="agentpool",
                user_type="external",
                is_sidechain=False,
            )

        # Assistant message
        content_blocks = [ClaudeMessageContent(type="text", text=message.content)]
        usage = ClaudeUsage()
        if message.cost_info:
            usage = ClaudeUsage(
                input_tokens=message.cost_info.token_usage.input_tokens,
                output_tokens=message.cost_info.token_usage.output_tokens,
            )
        assistant_msg = ClaudeApiMessage(
            model=message.model_name or "unknown",
            id=f"msg_{msg_uuid[:20]}",
            role="assistant",
            content=content_blocks,
            usage=usage,
        )
        return ClaudeAssistantEntry(
            type="assistant",
            uuid=msg_uuid,
            parent_uuid=parent_uuid,
            sessionId=session_id,
            timestamp=timestamp,
            message=assistant_msg,
            cwd=cwd or "",
            version="agentpool",
            user_type="external",
            is_sidechain=False,
        )

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages: list[ChatMessage[str]] = []

        # Determine which sessions to search
        sessions = self._list_sessions()

        for session_id, session_path in sessions:
            # Filter by conversation/session name if specified
            if query.name and session_id != query.name:
                continue

            entries = self._read_session(session_path)
            tool_mapping = self._build_tool_id_mapping(entries)

            for entry in entries:
                msg = self._entry_to_chat_message(entry, session_id, tool_mapping)
                if msg is None:
                    continue

                # Apply filters
                if query.agents and msg.name not in query.agents:
                    continue

                cutoff = query.get_time_cutoff()
                if query.since and cutoff and msg.timestamp and msg.timestamp < cutoff:
                    continue

                if query.until and msg.timestamp:
                    until_dt = datetime.fromisoformat(query.until)
                    if msg.timestamp > until_dt:
                        continue

                if query.contains and query.contains not in msg.content:
                    continue

                if query.roles and msg.role not in query.roles:
                    continue

                messages.append(msg)

                if query.limit and len(messages) >= query.limit:
                    return messages

        return messages

    async def log_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        parent_id: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        provider_name: str | None = None,
        provider_response_id: str | None = None,
        messages: str | None = None,
        finish_reason: FinishReason | None = None,
    ) -> None:
        """Log a message to Claude format.

        Note: conversation_id should be in format "project_path:session_id"
        or just "session_id" (will use default project).
        """
        # Parse conversation_id
        if ":" in conversation_id:
            project_path, session_id = conversation_id.split(":", 1)
        else:
            project_path = "/tmp"
            session_id = conversation_id

        # Build ChatMessage for conversion
        chat_message = ChatMessage[str](
            content=content,
            conversation_id=conversation_id,
            role=cast(MessageRole, role),
            message_id=message_id,
            name=name,
            model_name=model,
            cost_info=cost_info,
            response_time=response_time,
            parent_id=parent_id,
        )

        # Convert to entry and write
        entry = self._chat_message_to_entry(
            chat_message,
            session_id=session_id,
            parent_uuid=parent_id,
            cwd=project_path,
        )

        session_path = self._get_project_dir(project_path) / f"{session_id}.jsonl"
        self._write_entry(session_path, entry)

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log a conversation start.

        In Claude format, conversations are implicit (created when first message is written).
        This is a no-op but could be extended to create an initial entry.
        """

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations with their messages."""
        from agentpool_storage.models import ConversationData as ConvData

        result: list[tuple[ConvData, Sequence[ChatMessage[str]]]] = []
        sessions = self._list_sessions()

        for session_id, session_path in sessions:
            entries = self._read_session(session_path)
            if not entries:
                continue

            tool_mapping = self._build_tool_id_mapping(entries)

            # Build messages
            messages: list[ChatMessage[str]] = []
            first_timestamp: datetime | None = None
            total_tokens = 0

            for entry in entries:
                msg = self._entry_to_chat_message(entry, session_id, tool_mapping)
                if msg is None:
                    continue

                messages.append(msg)

                if first_timestamp is None and msg.timestamp:
                    first_timestamp = msg.timestamp

                if msg.cost_info:
                    total_tokens += msg.cost_info.token_usage.total_tokens

            if not messages:
                continue

            # Apply filters
            if filters.agent_name and not any(m.name == filters.agent_name for m in messages):
                continue

            if filters.since and first_timestamp and first_timestamp < filters.since:
                continue

            if filters.query and not any(filters.query in m.content for m in messages):
                continue

            # Build MessageData list
            msg_data_list: list[MessageData] = []
            for msg in messages:
                msg_data: MessageData = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": (msg.timestamp or get_now()).isoformat(),
                    "parent_id": msg.parent_id,
                    "model": msg.model_name,
                    "name": msg.name,
                    "token_usage": TokenUsage(
                        total=msg.cost_info.token_usage.total_tokens if msg.cost_info else 0,
                        prompt=msg.cost_info.token_usage.input_tokens if msg.cost_info else 0,
                        completion=msg.cost_info.token_usage.output_tokens if msg.cost_info else 0,
                    )
                    if msg.cost_info
                    else None,
                    "cost": float(msg.cost_info.total_cost) if msg.cost_info else None,
                    "response_time": msg.response_time,
                }
                msg_data_list.append(msg_data)

            token_usage_data: TokenUsage | None = (
                {"total": total_tokens, "prompt": 0, "completion": 0} if total_tokens else None
            )
            conv_data = ConvData(
                id=session_id,
                agent=messages[0].name or "claude",
                title=None,
                start_time=(first_timestamp or get_now()).isoformat(),
                messages=msg_data_list,
                token_usage=token_usage_data,
            )

            result.append((conv_data, messages))

            if filters.limit and len(result) >= filters.limit:
                break

        return result

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get conversation statistics."""
        from collections import defaultdict

        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_tokens": 0, "messages": 0, "models": set()}
        )

        sessions = self._list_sessions()

        for _session_id, session_path in sessions:
            entries = self._read_session(session_path)

            for entry in entries:
                if not isinstance(entry, ClaudeAssistantEntry):
                    continue

                if not isinstance(entry.message, ClaudeApiMessage):
                    continue

                api_msg = entry.message
                model = api_msg.model
                usage = api_msg.usage
                total_tokens = (
                    usage.input_tokens + usage.output_tokens + usage.cache_read_input_tokens
                )

                try:
                    timestamp = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    timestamp = get_now()

                # Apply time filter
                if timestamp < filters.cutoff:
                    continue

                # Group by specified criterion
                match filters.group_by:
                    case "model":
                        key = model
                    case "hour":
                        key = timestamp.strftime("%Y-%m-%d %H:00")
                    case "day":
                        key = timestamp.strftime("%Y-%m-%d")
                    case _:
                        key = "claude"  # Default agent grouping

                stats[key]["messages"] += 1
                stats[key]["total_tokens"] += total_tokens
                stats[key]["models"].add(model)

        # Convert sets to lists for JSON serialization
        for value in stats.values():
            value["models"] = list(value["models"])

        return dict(stats)

    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        """Reset storage.

        Warning: This will delete Claude conversation files!
        """
        conv_count = 0
        msg_count = 0

        sessions = self._list_sessions()

        for _session_id, session_path in sessions:
            entries = self._read_session(session_path)
            msg_count += len([
                e for e in entries if isinstance(e, (ClaudeUserEntry, ClaudeAssistantEntry))
            ])
            conv_count += 1

            if hard or not agent_name:
                session_path.unlink(missing_ok=True)

        return conv_count, msg_count

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        conv_count = 0
        msg_count = 0

        sessions = self._list_sessions()

        for _session_id, session_path in sessions:
            entries = self._read_session(session_path)
            message_entries = [
                e for e in entries if isinstance(e, (ClaudeUserEntry, ClaudeAssistantEntry))
            ]

            if message_entries:
                conv_count += 1
                msg_count += len(message_entries)

        return conv_count, msg_count

    async def get_conversation_messages(
        self,
        conversation_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a conversation.

        Args:
            conversation_id: Session ID (conversation ID in Claude format)
            include_ancestors: If True, traverse parent_uuid chain to include
                messages from ancestor conversations

        Returns:
            List of messages ordered by timestamp
        """
        # Find the session file
        sessions = self._list_sessions()
        session_path = None
        for sid, spath in sessions:
            if sid == conversation_id:
                session_path = spath
                break

        if not session_path:
            return []

        # Read entries and convert to messages
        entries = self._read_session(session_path)
        tool_mapping = self._build_tool_id_mapping(entries)

        messages: list[ChatMessage[str]] = []
        for entry in entries:
            msg = self._entry_to_chat_message(entry, conversation_id, tool_mapping)
            if msg:
                messages.append(msg)

        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp or get_now())

        if not include_ancestors or not messages:
            return messages

        # Get ancestor chain if first message has parent_id
        first_msg = messages[0]
        if first_msg.parent_id:
            ancestors = await self.get_message_ancestry(first_msg.parent_id)
            return ancestors + messages

        return messages

    async def get_message(self, message_id: str) -> ChatMessage[str] | None:
        """Get a single message by ID.

        Args:
            message_id: UUID of the message

        Returns:
            The message if found, None otherwise
        """
        # Search all sessions for the message
        sessions = self._list_sessions()

        for session_id, session_path in sessions:
            entries = self._read_session(session_path)
            tool_mapping = self._build_tool_id_mapping(entries)

            for entry in entries:
                if (
                    isinstance(entry, (ClaudeUserEntry, ClaudeAssistantEntry))
                    and entry.uuid == message_id
                ):
                    return self._entry_to_chat_message(entry, session_id, tool_mapping)

        return None

    async def get_message_ancestry(self, message_id: str) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Traverses parent_uuid chain to build full history.

        Args:
            message_id: UUID of the message

        Returns:
            List of messages from oldest ancestor to the specified message
        """
        ancestors: list[ChatMessage[str]] = []
        current_id: str | None = message_id

        while current_id:
            msg = await self.get_message(current_id)
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
        source_conversation_id: str,
        new_conversation_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        """Fork a conversation at a specific point.

        Creates a new session file. The fork point message_id is returned
        so callers can set it as parent_uuid for new messages.

        Args:
            source_conversation_id: Source session ID
            new_conversation_id: New session ID
            fork_from_message_id: UUID to fork from. If None, forks from last message
            new_agent_name: Not used in Claude format (no agent metadata in sessions)

        Returns:
            The UUID of the fork point message
        """
        # Find source session
        sessions = self._list_sessions()
        source_path = None
        for sid, spath in sessions:
            if sid == source_conversation_id:
                source_path = spath
                break

        if not source_path:
            msg = f"Source conversation not found: {source_conversation_id}"
            raise ValueError(msg)

        # Read source entries
        entries = self._read_session(source_path)

        # Find fork point
        fork_point_id: str | None = None
        if fork_from_message_id:
            # Verify message exists
            found = False
            for entry in entries:
                if (
                    isinstance(entry, (ClaudeUserEntry, ClaudeAssistantEntry))
                    and entry.uuid == fork_from_message_id
                ):
                    found = True
                    fork_point_id = fork_from_message_id
                    break
            if not found:
                err = f"Message {fork_from_message_id} not found in conversation"
                raise ValueError(err)
        else:
            # Find last message
            message_entries = [
                e for e in entries if isinstance(e, (ClaudeUserEntry, ClaudeAssistantEntry))
            ]
            if message_entries:
                fork_point_id = message_entries[-1].uuid

        # Create new session file (empty for now - will be populated when messages added)
        # Determine project from source path structure
        project_name = source_path.parent.name
        new_path = self.projects_path / project_name / f"{new_conversation_id}.jsonl"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.touch()

        return fork_point_id
