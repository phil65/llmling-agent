"""OpenCode storage provider - reads/writes to ~/.local/share/opencode format."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import anyenv
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
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

from agentpool.log import get_logger
from agentpool.messaging import ChatMessage, TokenCost
from agentpool.utils.now import get_now
from agentpool_storage.base import StorageProvider
from agentpool_storage.models import TokenUsage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentpool_config.session import SessionQuery
    from agentpool_config.storage import OpenCodeStorageConfig
    from agentpool_storage.models import (
        ConversationData,
        MessageData,
        QueryFilters,
        StatsFilters,
    )

logger = get_logger(__name__)

# OpenCode version we're emulating
OPENCODE_VERSION = "1.1.7"

# OpenCode data models

PartType = Literal["text", "step-start", "step-finish", "reasoning", "tool", "patch", "compaction"]
ToolStatus = Literal["pending", "running", "completed", "error"]
FinishReason = Literal["stop", "tool-calls", "length", "error"]


class OpenCodeTime(BaseModel):
    """Timestamp fields used in OpenCode."""

    created: int  # Unix timestamp in milliseconds
    updated: int | None = None
    completed: int | None = None


class OpenCodeSummary(BaseModel):
    """Summary information for sessions/messages."""

    additions: int | None = None
    deletions: int | None = None
    files: int = 0
    title: str | None = None
    diffs: list[Any] = Field(default_factory=list)


class OpenCodeModel(BaseModel):
    """Model information in messages."""

    provider_id: str = Field(alias="providerID")
    model_id: str = Field(alias="modelID")

    model_config = ConfigDict(populate_by_name=True)


class OpenCodePath(BaseModel):
    """Path context in messages."""

    cwd: str
    root: str


class OpenCodeTokens(BaseModel):
    """Token usage information."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache: dict[str, int] = Field(default_factory=dict)


class OpenCodeSession(BaseModel):
    """OpenCode session metadata."""

    id: str
    version: str = OPENCODE_VERSION
    project_id: str = Field(alias="projectID")
    directory: str
    title: str
    time: OpenCodeTime
    summary: OpenCodeSummary = Field(default_factory=OpenCodeSummary)

    model_config = ConfigDict(populate_by_name=True)


class OpenCodeMessage(BaseModel):
    """OpenCode message metadata."""

    id: str
    session_id: str = Field(alias="sessionID")
    role: Literal["user", "assistant"]
    time: OpenCodeTime
    summary: OpenCodeSummary | bool | None = None
    agent: str | None = None
    model: OpenCodeModel | None = None
    parent_id: str | None = Field(default=None, alias="parentID")
    model_id: str | None = Field(default=None, alias="modelID")
    provider_id: str | None = Field(default=None, alias="providerID")
    mode: str | None = None
    path: OpenCodePath | None = None
    cost: float | None = None
    tokens: OpenCodeTokens | None = None
    finish: FinishReason | None = None

    model_config = ConfigDict(populate_by_name=True)


class OpenCodeToolState(BaseModel):
    """Tool execution state."""

    status: ToolStatus
    input: dict[str, Any] | None = None
    output: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None
    time: dict[str, int] | None = None


class OpenCodePartBase(BaseModel):
    """Base for all part types."""

    id: str
    session_id: str = Field(alias="sessionID")
    message_id: str = Field(alias="messageID")
    type: PartType

    model_config = ConfigDict(populate_by_name=True)


class OpenCodeTextPart(OpenCodePartBase):
    """Text content part."""

    type: Literal["text"]
    text: str
    time: dict[str, int] | None = None


class OpenCodeReasoningPart(OpenCodePartBase):
    """Reasoning/thinking content part."""

    type: Literal["reasoning"]
    text: str
    time: dict[str, int] | None = None


class OpenCodeToolPart(OpenCodePartBase):
    """Tool call/result part."""

    type: Literal["tool"]
    call_id: str = Field(alias="callID")
    tool: str
    state: OpenCodeToolState


class OpenCodeStepStartPart(OpenCodePartBase):
    """Step start marker."""

    type: Literal["step-start"]


class OpenCodeStepFinishPart(OpenCodePartBase):
    """Step finish marker with stats."""

    type: Literal["step-finish"]
    reason: FinishReason | None = None
    cost: float | None = None
    tokens: OpenCodeTokens | None = None


class OpenCodePatchPart(OpenCodePartBase):
    """File patch/diff part."""

    type: Literal["patch"]
    hash: str | None = None
    files: list[str] = Field(default_factory=list)


class OpenCodeCompactionPart(OpenCodePartBase):
    """Compaction marker."""

    type: Literal["compaction"]


# Discriminated union for all part types
OpenCodePart = Annotated[
    OpenCodeTextPart
    | OpenCodeReasoningPart
    | OpenCodeToolPart
    | OpenCodeStepStartPart
    | OpenCodeStepFinishPart
    | OpenCodePatchPart
    | OpenCodeCompactionPart,
    Field(discriminator="type"),
]


class OpenCodeStorageProvider(StorageProvider):
    """Storage provider that reads/writes OpenCode's native format.

    OpenCode stores data in:
    - ~/.local/share/opencode/storage/session/{project_id}/ - Session JSON files
    - ~/.local/share/opencode/storage/message/{session_id}/ - Message JSON files
    - ~/.local/share/opencode/storage/part/{message_id}/ - Part JSON files

    Each file is a single JSON object (not JSONL).
    """

    can_load_history = True

    def __init__(self, config: OpenCodeStorageConfig) -> None:
        """Initialize OpenCode storage provider."""
        super().__init__(config)
        self.base_path = Path(config.path).expanduser()
        self.sessions_path = self.base_path / "session"
        self.messages_path = self.base_path / "message"
        self.parts_path = self.base_path / "part"

    def _ms_to_datetime(self, ms: int) -> datetime:
        """Convert milliseconds timestamp to datetime."""
        return datetime.fromtimestamp(ms / 1000, tz=UTC)

    def _list_sessions(self, project_id: str | None = None) -> list[tuple[str, Path]]:
        """List all sessions, optionally filtered by project."""
        sessions: list[tuple[str, Path]] = []
        if not self.sessions_path.exists():
            return sessions

        if project_id:
            project_dir = self.sessions_path / project_id
            if project_dir.exists():
                sessions.extend((f.stem, f) for f in project_dir.glob("*.json"))
        else:
            for project_dir in self.sessions_path.iterdir():
                if project_dir.is_dir():
                    sessions.extend((f.stem, f) for f in project_dir.glob("*.json"))
        return sessions

    def _read_session(self, session_path: Path) -> OpenCodeSession | None:
        """Read session metadata."""
        if not session_path.exists():
            return None
        try:
            content = session_path.read_text(encoding="utf-8")
            return anyenv.load_json(content, return_type=OpenCodeSession)
        except anyenv.JsonLoadError as e:
            logger.warning("Failed to parse session", path=str(session_path), error=str(e))
            return None

    def _read_messages(self, session_id: str) -> list[OpenCodeMessage]:
        """Read all messages for a session."""
        messages: list[OpenCodeMessage] = []
        msg_dir = self.messages_path / session_id
        if not msg_dir.exists():
            return messages

        for msg_file in sorted(msg_dir.glob("*.json")):
            try:
                content = msg_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content, return_type=OpenCodeMessage)
                messages.append(data)
            except anyenv.JsonLoadError as e:
                logger.warning("Failed to parse message", path=str(msg_file), error=str(e))
        return messages

    def _read_parts(self, message_id: str) -> list[OpenCodePart]:
        """Read all parts for a message."""
        parts: list[OpenCodePart] = []
        parts_dir = self.parts_path / message_id
        if not parts_dir.exists():
            return parts

        adapter = TypeAdapter[Any](OpenCodePart)
        for part_file in sorted(parts_dir.glob("*.json")):
            try:
                content = part_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content)
                parts.append(adapter.validate_python(data))
            except anyenv.JsonLoadError as e:
                logger.warning("Failed to parse part", path=str(part_file), error=str(e))
        return parts

    def _build_tool_id_mapping(self, parts: list[OpenCodePart]) -> dict[str, str]:
        """Build mapping from tool callID to tool name."""
        mapping: dict[str, str] = {}
        for part in parts:
            if isinstance(part, OpenCodeToolPart):
                mapping[part.call_id] = part.tool
        return mapping

    def _message_to_chat_message(
        self,
        msg: OpenCodeMessage,
        parts: list[OpenCodePart],
        conversation_id: str,
        tool_id_mapping: dict[str, str] | None = None,
    ) -> ChatMessage[str]:
        """Convert OpenCode message + parts to ChatMessage."""
        timestamp = self._ms_to_datetime(msg.time.created)

        # Extract text content for display
        content = self._extract_text_content(parts)

        # Build pydantic-ai messages
        pydantic_messages = self._build_pydantic_messages(
            msg, parts, timestamp, tool_id_mapping or {}
        )

        # Extract cost info
        cost_info = None
        if msg.tokens:
            input_tokens = msg.tokens.input + msg.tokens.cache.get("read", 0)
            output_tokens = msg.tokens.output
            if input_tokens or output_tokens:
                cost_info = TokenCost(
                    token_usage=RunUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                    total_cost=Decimal(str(msg.cost)) if msg.cost else Decimal(0),
                )

        # Get model name
        model_name = msg.model_id
        if not model_name and msg.model:
            model_name = msg.model.model_id

        return ChatMessage[str](
            content=content,
            conversation_id=conversation_id,
            role=msg.role,
            message_id=msg.id,
            name=msg.agent,
            model_name=model_name,
            cost_info=cost_info,
            timestamp=timestamp,
            parent_id=msg.parent_id,
            messages=pydantic_messages,
            provider_details={"finish_reason": msg.finish} if msg.finish else {},
        )

    def _extract_text_content(self, parts: list[OpenCodePart]) -> str:
        """Extract text content from parts for display."""
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, OpenCodeTextPart) and part.text:
                text_parts.append(part.text)
            elif isinstance(part, OpenCodeReasoningPart) and part.text:
                text_parts.append(f"<thinking>\n{part.text}\n</thinking>")
        return "\n".join(text_parts)

    def _build_pydantic_messages(
        self,
        msg: OpenCodeMessage,
        parts: list[OpenCodePart],
        timestamp: datetime,
        tool_id_mapping: dict[str, str],
    ) -> list[ModelRequest | ModelResponse]:
        """Build pydantic-ai ModelRequest and/or ModelResponse.

        In OpenCode's model, assistant messages contain both tool calls AND their
        results in the same message. We split these into:
        - ModelResponse with ToolCallPart (the call)
        - ModelRequest with ToolReturnPart (the result)
        """
        result: list[ModelRequest | ModelResponse] = []

        if msg.role == "user":
            request_parts: list[UserPromptPart | ToolReturnPart] = [
                UserPromptPart(content=part.text, timestamp=timestamp)
                for part in parts
                if isinstance(part, OpenCodeTextPart) and part.text
            ]
            if request_parts:
                result.append(ModelRequest(parts=request_parts, timestamp=timestamp))
            return result

        # Assistant message - may contain both tool calls and results
        response_parts: list[TextPart | ToolCallPart | ThinkingPart] = []
        tool_return_parts: list[ToolReturnPart] = []

        # Build usage
        usage = RequestUsage()
        if msg.tokens:
            usage = RequestUsage(
                input_tokens=msg.tokens.input,
                output_tokens=msg.tokens.output,
                cache_read_tokens=msg.tokens.cache.get("read", 0),
                cache_write_tokens=msg.tokens.cache.get("write", 0),
            )

        for part in parts:
            if isinstance(part, OpenCodeTextPart) and part.text:
                response_parts.append(TextPart(content=part.text))
            elif isinstance(part, OpenCodeReasoningPart) and part.text:
                response_parts.append(ThinkingPart(content=part.text))
            elif isinstance(part, OpenCodeToolPart):
                # Add tool call to response
                args = part.state.input or {}
                tc_part = ToolCallPart(tool_name=part.tool, args=args, tool_call_id=part.call_id)
                response_parts.append(tc_part)
                # If completed, also create a tool return
                if part.state.status == "completed" and part.state.output:
                    return_part = ToolReturnPart(
                        tool_name=part.tool,
                        content=part.state.output,
                        tool_call_id=part.call_id,
                        timestamp=timestamp,
                    )
                    tool_return_parts.append(return_part)

        # Add the response if we have parts
        if response_parts:
            model_name = msg.model_id or (msg.model.model_id if msg.model else None)
            result.append(
                ModelResponse(
                    parts=response_parts,
                    usage=usage,
                    model_name=model_name,
                    timestamp=timestamp,
                )
            )

        # Add tool returns as a separate request (simulating user sending results back)
        if tool_return_parts:
            result.append(ModelRequest(parts=tool_return_parts, timestamp=timestamp))

        return result

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages: list[ChatMessage[str]] = []
        sessions = self._list_sessions()

        for session_id, session_path in sessions:
            if query.name and session_id != query.name:
                continue

            session = self._read_session(session_path)
            if not session:
                continue

            oc_messages = self._read_messages(session_id)

            # Build tool mapping from all parts
            all_parts: list[OpenCodePart] = []
            msg_parts_map: dict[str, list[OpenCodePart]] = {}
            for oc_msg in oc_messages:
                parts = self._read_parts(oc_msg.id)
                msg_parts_map[oc_msg.id] = parts
                all_parts.extend(parts)
            tool_mapping = self._build_tool_id_mapping(all_parts)

            for oc_msg in oc_messages:
                parts = msg_parts_map.get(oc_msg.id, [])
                chat_msg = self._message_to_chat_message(oc_msg, parts, session_id, tool_mapping)

                # Apply filters
                if query.agents and chat_msg.name not in query.agents:
                    continue
                cutoff = query.get_time_cutoff()
                if query.since and cutoff and chat_msg.timestamp < cutoff:
                    continue
                if query.until:
                    until_dt = datetime.fromisoformat(query.until)
                    if chat_msg.timestamp > until_dt:
                        continue
                if query.contains and query.contains not in chat_msg.content:
                    continue
                if query.roles and chat_msg.role not in query.roles:
                    continue
                messages.append(chat_msg)

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
        finish_reason: Any | None = None,
    ) -> None:
        """Log a message to OpenCode format.

        Note: Writing to OpenCode format is not fully implemented.
        """
        logger.warning("Writing to OpenCode format is not fully supported")

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log a conversation start."""
        # No-op for read-only provider

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations with their messages."""
        from agentpool_storage.models import ConversationData as ConvData

        result: list[tuple[ConvData, Sequence[ChatMessage[str]]]] = []
        sessions = self._list_sessions()
        for session_id, session_path in sessions:
            session = self._read_session(session_path)
            if not session:
                continue

            oc_messages = self._read_messages(session_id)
            if not oc_messages:
                continue

            # Build tool mapping
            all_parts: list[OpenCodePart] = []
            msg_parts_map: dict[str, list[OpenCodePart]] = {}
            for oc_msg in oc_messages:
                parts = self._read_parts(oc_msg.id)
                msg_parts_map[oc_msg.id] = parts
                all_parts.extend(parts)
            tool_mapping = self._build_tool_id_mapping(all_parts)

            # Convert messages
            chat_messages: list[ChatMessage[str]] = []
            total_tokens = 0
            total_cost = 0.0

            for oc_msg in oc_messages:
                parts = msg_parts_map.get(oc_msg.id, [])
                chat_msg = self._message_to_chat_message(oc_msg, parts, session_id, tool_mapping)
                chat_messages.append(chat_msg)

                if oc_msg.tokens:
                    total_tokens += oc_msg.tokens.input + oc_msg.tokens.output
                if oc_msg.cost:
                    total_cost += oc_msg.cost

            if not chat_messages:
                continue

            first_timestamp = self._ms_to_datetime(session.time.created)

            # Apply filters
            if filters.agent_name and not any(m.name == filters.agent_name for m in chat_messages):
                continue

            if filters.since and first_timestamp < filters.since:
                continue

            if filters.query and not any(filters.query in m.content for m in chat_messages):
                continue

            # Build MessageData list
            msg_data_list: list[MessageData] = []
            for chat_msg in chat_messages:
                msg_data: MessageData = {
                    "role": chat_msg.role,
                    "content": chat_msg.content,
                    "timestamp": (chat_msg.timestamp or get_now()).isoformat(),
                    "parent_id": chat_msg.parent_id,
                    "model": chat_msg.model_name,
                    "name": chat_msg.name,
                    "token_usage": TokenUsage(
                        total=chat_msg.cost_info.token_usage.total_tokens
                        if chat_msg.cost_info
                        else 0,
                        prompt=chat_msg.cost_info.token_usage.input_tokens
                        if chat_msg.cost_info
                        else 0,
                        completion=chat_msg.cost_info.token_usage.output_tokens
                        if chat_msg.cost_info
                        else 0,
                    )
                    if chat_msg.cost_info
                    else None,
                    "cost": float(chat_msg.cost_info.total_cost) if chat_msg.cost_info else None,
                    "response_time": chat_msg.response_time,
                }
                msg_data_list.append(msg_data)

            token_usage_data: TokenUsage | None = (
                {"total": total_tokens, "prompt": 0, "completion": 0} if total_tokens else None
            )
            conv_data = ConvData(
                id=session_id,
                agent=chat_messages[0].name or "opencode",
                title=session.title,
                start_time=first_timestamp.isoformat(),
                messages=msg_data_list,
                token_usage=token_usage_data,
            )

            result.append((conv_data, chat_messages))

            if filters.limit and len(result) >= filters.limit:
                break

        return result

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get conversation statistics."""
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_tokens": 0, "messages": 0, "models": set(), "total_cost": 0.0}
        )

        sessions = self._list_sessions()

        for _session_id, session_path in sessions:
            session = self._read_session(session_path)
            if not session:
                continue

            timestamp = self._ms_to_datetime(session.time.created)
            if timestamp < filters.cutoff:
                continue

            oc_messages = self._read_messages(session.id)

            for oc_msg in oc_messages:
                if oc_msg.role != "assistant":
                    continue

                model = oc_msg.model_id or (oc_msg.model.model_id if oc_msg.model else "unknown")
                tokens = 0
                if oc_msg.tokens:
                    tokens = oc_msg.tokens.input + oc_msg.tokens.output

                msg_timestamp = self._ms_to_datetime(oc_msg.time.created)

                # Group by specified criterion
                match filters.group_by:
                    case "model":
                        key = model
                    case "hour":
                        key = msg_timestamp.strftime("%Y-%m-%d %H:00")
                    case "day":
                        key = msg_timestamp.strftime("%Y-%m-%d")
                    case _:
                        key = oc_msg.agent or "opencode"

                stats[key]["messages"] += 1
                stats[key]["total_tokens"] += tokens
                stats[key]["models"].add(model)
                stats[key]["total_cost"] += oc_msg.cost or 0.0

        # Convert sets to lists
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

        Warning: This would delete OpenCode data!
        """
        logger.warning("Reset not implemented for OpenCode storage (read-only)")
        return 0, 0

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        conv_count = 0
        msg_count = 0

        sessions = self._list_sessions()

        for session_id, session_path in sessions:
            session = self._read_session(session_path)
            if not session:
                continue

            oc_messages = self._read_messages(session_id)
            if oc_messages:
                conv_count += 1
                msg_count += len(oc_messages)

        return conv_count, msg_count

    async def get_conversation_messages(
        self,
        conversation_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a conversation.

        Args:
            conversation_id: Session ID (conversation ID in OpenCode format)
            include_ancestors: If True, traverse parent_id chain to include
                messages from ancestor conversations

        Returns:
            List of messages ordered by timestamp
        """
        # Read messages for this session
        oc_messages = self._read_messages(conversation_id)

        messages: list[ChatMessage[str]] = []
        for oc_msg in oc_messages:
            parts = self._read_parts(oc_msg.id)
            tool_mapping = self._build_tool_id_mapping(parts)
            chat_msg = self._message_to_chat_message(oc_msg, parts, conversation_id, tool_mapping)
            messages.append(chat_msg)

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
            message_id: ID of the message

        Returns:
            The message if found, None otherwise
        """
        # Search all sessions for the message
        sessions = self._list_sessions()

        for session_id, _session_path in sessions:
            oc_messages = self._read_messages(session_id)
            for oc_msg in oc_messages:
                if oc_msg.id == message_id:
                    parts = self._read_parts(oc_msg.id)
                    tool_mapping = self._build_tool_id_mapping(parts)
                    return self._message_to_chat_message(oc_msg, parts, session_id, tool_mapping)

        return None

    async def get_message_ancestry(self, message_id: str) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Traverses parent_id chain to build full history.

        Args:
            message_id: ID of the message

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

        Creates a new session directory. The fork point message_id is returned
        so callers can set it as parent_id for new messages.

        Args:
            source_conversation_id: Source session ID
            new_conversation_id: New session ID
            fork_from_message_id: Message ID to fork from. If None, forks from last
            new_agent_name: Not directly stored in OpenCode session format

        Returns:
            The ID of the fork point message
        """
        # Find source session
        sessions = self._list_sessions()
        source_session = None
        source_path = None
        for session_id, session_path in sessions:
            if session_id == source_conversation_id:
                source_session = self._read_session(session_path)
                source_path = session_path
                break

        if not source_session or not source_path:
            msg = f"Source conversation not found: {source_conversation_id}"
            raise ValueError(msg)

        # Read source messages
        oc_messages = self._read_messages(source_conversation_id)

        # Find fork point
        fork_point_id: str | None = None
        if fork_from_message_id:
            # Verify message exists
            found = any(m.id == fork_from_message_id for m in oc_messages)
            if not found:
                err = f"Message {fork_from_message_id} not found in conversation"
                raise ValueError(err)
            fork_point_id = fork_from_message_id
        # Fork from last message
        elif oc_messages:
            # Messages are already in time order from _read_messages
            fork_point_id = oc_messages[-1].id

        # Create new session directory structure
        # Determine project from source path structure
        project_id = source_path.parent.name
        new_session_dir = self.sessions_path / project_id
        new_session_dir.mkdir(parents=True, exist_ok=True)

        # Create empty session file (will be populated when messages added)
        new_session_path = new_session_dir / f"{new_conversation_id}.json"

        # Create new session metadata
        new_session = OpenCodeSession(
            id=new_conversation_id,
            project_id=project_id,
            time=OpenCodeTime(
                created=int(get_now().timestamp() * 1000),
            ),
            status="active",
            summary=OpenCodeSummary(
                title=f"{source_session.summary.title or 'Session'} (fork)"
                if source_session.summary.title
                else None,
            ),
        )

        # Write session file
        new_session_path.write_text(
            anyenv.dump_json(new_session.model_dump(by_alias=True), indent=True),
            encoding="utf-8",
        )

        # Create message and part directories
        (self.messages_path / new_conversation_id).mkdir(parents=True, exist_ok=True)
        (self.parts_path / new_conversation_id).mkdir(parents=True, exist_ok=True)

        return fork_point_id


if __name__ == "__main__":
    import asyncio
    import datetime as dt

    from agentpool_config.storage import OpenCodeStorageConfig
    from agentpool_storage.models import QueryFilters, StatsFilters

    async def main() -> None:
        config = OpenCodeStorageConfig()
        provider = OpenCodeStorageProvider(config)

        print(f"Base path: {provider.base_path}")
        print(f"Exists: {provider.base_path.exists()}")

        # List conversations
        filters = QueryFilters(limit=10)
        conversations = await provider.get_conversations(filters)
        print(f"\nFound {len(conversations)} conversations")

        for conv_data, messages in conversations[:5]:
            print(f"  - {conv_data['id'][:8]}... | {conv_data['title'] or 'Untitled'}")
            print(f"    Messages: {len(messages)}, Updated: {conv_data['start_time']}")

        # Get counts
        conv_count, msg_count = await provider.get_conversation_counts()
        print(f"\nTotal: {conv_count} conversations, {msg_count} messages")

        # Get stats
        stats_filters = StatsFilters(
            cutoff=dt.datetime.now(dt.UTC) - dt.timedelta(days=30),
            group_by="day",
        )
        stats = await provider.get_conversation_stats(stats_filters)
        print(f"\nStats: {stats}")

    asyncio.run(main())
