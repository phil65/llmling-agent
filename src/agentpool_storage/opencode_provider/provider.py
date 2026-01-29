"""OpenCode storage provider.

This module implements storage compatible with OpenCode's normalized JSON format,
storing conversations as relational data across multiple directories.

Key differences from Claude Code:
- Normalized structure (sessions → messages → parts)
- SHA1-based project IDs
- Timestamp-based message ordering (no parent links)
- In-place file updates (not append-only)

See ARCHITECTURE.md for detailed documentation of the storage format.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anyenv
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentpool.log import get_logger
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict
from agentpool.utils.time_utils import datetime_to_ms, get_now, ms_to_datetime
from agentpool_server.opencode_server.models import (
    AssistantMessage,
    MessageInfo as OpenCodeMessage,
    Part as OpenCodePart,
    Session,
    SessionSummary,
    TimeCreatedUpdated,
)
from agentpool_storage.base import StorageProvider
from agentpool_storage.models import ConversationData as ConvData, TokenUsage
from agentpool_storage.opencode_provider import helpers


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentpool.messaging import ChatMessage, TokenCost
    from agentpool_config.session import SessionQuery
    from agentpool_config.storage import OpenCodeStorageConfig
    from agentpool_storage.models import ConversationData, MessageData, QueryFilters, StatsFilters

logger = get_logger(__name__)

# OpenCode version we're emulating
OPENCODE_VERSION = "1.1.7"

# Type aliases - use server models directly
PartType = Literal[
    "text",
    "step-start",
    "step-finish",
    "reasoning",
    "tool",
    "patch",
    "compaction",
    "snapshot",
    "agent",
    "subtask",
    "retry",
]
ToolStatus = Literal["pending", "running", "completed", "error"]
FinishReason = Literal["stop", "tool-calls", "length", "error"]


@dataclass(slots=True)
class OpenCodeSessionMetadata:
    """Lightweight session metadata without loading messages/parts."""

    session_id: str
    path: Path
    title: str
    directory: str
    created_at: datetime
    updated_at: datetime
    project_id: str


def _read_session_metadata(session_path: Path) -> OpenCodeSessionMetadata | None:
    """Read minimal metadata from a session file without loading messages/parts.

    Args:
        session_path: Path to the session JSON file

    Returns:
        OpenCodeSessionMetadata or None if file is invalid
    """
    session = helpers.read_session(session_path)
    if session is None:
        return None

    return OpenCodeSessionMetadata(
        session_id=session.id,
        path=session_path,
        title=session.title,
        directory=session.directory,
        created_at=ms_to_datetime(session.time.created),
        updated_at=ms_to_datetime(session.time.updated),
        project_id=session.project_id,
    )


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

    def _list_sessions(self, project_id: str | None = None) -> list[tuple[str, Path]]:
        """List all sessions, optionally filtered by project."""
        if not self.sessions_path.exists():
            return []
        sessions: list[tuple[str, Path]] = []
        if project_id:
            project_dir = self.sessions_path / project_id
            if project_dir.exists():
                sessions.extend((f.stem, f) for f in project_dir.glob("*.json"))
        else:
            for project_dir in self.sessions_path.iterdir():
                if project_dir.is_dir():
                    sessions.extend((f.stem, f) for f in project_dir.glob("*.json"))
        return sessions

    def list_session_metadata(
        self,
        project_id: str | None = None,
    ) -> list[OpenCodeSessionMetadata]:
        """List all sessions with lightweight metadata (no message/part loading).

        This is much faster than get_sessions() as it only reads session JSON files
        without loading messages or parts.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            List of OpenCodeSessionMetadata objects
        """
        result: list[OpenCodeSessionMetadata] = []
        for _, session_path in self._list_sessions(project_id=project_id):
            metadata = _read_session_metadata(session_path)
            if metadata is not None:
                result.append(metadata)
        return result

    def _read_messages(self, session_id: str) -> list[OpenCodeMessage]:
        """Read all messages for a session."""
        msg_dir = self.messages_path / session_id
        if not msg_dir.exists():
            return []
        messages: list[OpenCodeMessage] = []
        adapter = TypeAdapter[OpenCodeMessage](OpenCodeMessage)
        for msg_file in sorted(msg_dir.glob("*.json")):
            try:
                content = msg_file.read_text(encoding="utf-8")
                data_dict = anyenv.load_json(content)
                data = adapter.validate_python(data_dict)
                messages.append(data)
            except anyenv.JsonLoadError as e:
                logger.warning("Failed to parse message", path=str(msg_file), error=str(e))
        return messages

    def _read_parts(self, message_id: str) -> list[OpenCodePart]:
        """Read all parts for a message."""
        parts_dir = self.parts_path / message_id
        if not parts_dir.exists():
            return []

        parts: list[OpenCodePart] = []
        adapter = TypeAdapter[Any](OpenCodePart)
        for part_file in sorted(parts_dir.glob("*.json")):
            try:
                content = part_file.read_text(encoding="utf-8")
                data = anyenv.load_json(content)
                parts.append(adapter.validate_python(data))
            except anyenv.JsonLoadError as e:
                logger.warning("Failed to parse part", path=str(part_file), error=str(e))
        return parts

    async def _write_message(  # noqa: PLR0915
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        model_messages: list[ModelRequest | ModelResponse],
        parent_id: str | None = None,
        model: str | None = None,
        cost_info: TokenCost | None = None,
        finish_reason: Any | None = None,
    ) -> None:
        """Write a message in OpenCode format."""
        from agentpool_server.opencode_server.models import (
            AssistantMessage,
            MessagePath,
            MessageTime,
            ReasoningPart as OpenCodeReasoningPart,
            TextPart as OpenCodeTextPart,
            TimeCreated,
            TimeStartEndCompacted,
            TimeStartEndOptional,
            Tokens,
            TokensCache,
            ToolPart as OpenCodeToolPart,
            ToolStateCompleted,
            ToolStateError,
            ToolStatePending,
            ToolStateRunning,
            UserMessage,
            UserMessageModel,
        )

        now_ms = int(get_now().timestamp() * 1000)
        # Ensure message directory exists
        msg_dir = self.messages_path / session_id
        msg_dir.mkdir(parents=True, exist_ok=True)
        # Create OpenCode message based on role
        oc_message: OpenCodeMessage
        if role == "assistant":
            oc_message = AssistantMessage(
                id=message_id,
                session_id=session_id,
                parent_id=parent_id or "",
                model_id=model or "",
                provider_id="",  # TODO: get from somewhere
                path=MessagePath(cwd="", root=""),  # TODO: get real paths
                time=MessageTime(created=now_ms),
                tokens=Tokens(
                    input=cost_info.token_usage.input_tokens if cost_info else 0,
                    output=cost_info.token_usage.output_tokens if cost_info else 0,
                    cache=TokensCache(
                        read=cost_info.token_usage.cache_read_tokens if cost_info else 0,
                        write=cost_info.token_usage.cache_write_tokens if cost_info else 0,
                    ),
                )
                if cost_info
                else Tokens(),
                cost=float(cost_info.total_cost) if cost_info else 0.0,
                finish=finish_reason,
            )
        else:  # user message
            oc_message = UserMessage(
                id=message_id,
                session_id=session_id,
                time=TimeCreated(created=now_ms),
                model=UserMessageModel(provider_id="", model_id=model or "") if model else None,
            )

        # Write message file
        msg_file = msg_dir / f"{message_id}.json"
        dct = oc_message.model_dump(by_alias=True)
        msg_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")
        # Convert model messages to OpenCode parts
        parts_dir = self.parts_path / message_id
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_counter = 0
        for msg in model_messages:
            if isinstance(msg, ModelRequest):
                # User prompt parts
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        # Convert UserContent to OpenCode parts using helper
                        text_parts = helpers.convert_user_content_to_parts(
                            content=part.content,
                            message_id=message_id,
                            session_id=session_id,
                            part_counter_start=part_counter,
                        )
                        part_counter += len(text_parts)
                        # Write each part to disk
                        for text_part in text_parts:
                            part_file = parts_dir / f"{text_part.id}.json"
                            part_file.write_text(
                                anyenv.dump_json(text_part.model_dump(by_alias=True), indent=True),
                                encoding="utf-8",
                            )
                    elif isinstance(part, ToolReturnPart):
                        # Tool return - update existing tool part with output
                        tool_part_file = None
                        # Find the tool part with matching call_id
                        for existing_file in parts_dir.glob("*.json"):
                            try:
                                text = existing_file.read_text(encoding="utf-8")
                                content = anyenv.load_json(text, return_type=dict)
                                if (
                                    content.get("type") == "tool"
                                    and content.get("callID") == part.tool_call_id
                                ):
                                    tool_part_file = existing_file
                                    break
                            except Exception:  # noqa: BLE001
                                continue

                        if tool_part_file:
                            # Update the tool part with output - create new completed state
                            text = tool_part_file.read_text(encoding="utf-8")
                            tool_part = anyenv.load_json(text, return_type=OpenCodeToolPart)
                            # Create new ToolStateCompleted (states are immutable)
                            # All tool states have .input,
                            # but only Running/Completed/Error have .time
                            start_time = 0
                            if isinstance(
                                tool_part.state,
                                (ToolStateRunning, ToolStateCompleted, ToolStateError),
                            ):
                                start_time = tool_part.state.time.start

                            completed_state = ToolStateCompleted(
                                input=tool_part.state.input,
                                output=str(part.content),
                                title=tool_part.tool,
                                time=TimeStartEndCompacted(
                                    start=start_time,
                                    end=int(get_now().timestamp() * 1000),
                                ),
                            )
                            # Create new tool part with updated state
                            updated_tool_part = OpenCodeToolPart(
                                id=tool_part.id,
                                message_id=tool_part.message_id,
                                session_id=tool_part.session_id,
                                call_id=tool_part.call_id,
                                tool=tool_part.tool,
                                state=completed_state,
                            )
                            dct = updated_tool_part.model_dump(by_alias=True)
                            text = anyenv.dump_json(dct, indent=True)
                            tool_part_file.write_text(text, encoding="utf-8")

            elif isinstance(msg, ModelResponse):
                # Model response parts
                for part in msg.parts:  # type: ignore[assignment]
                    part_id = f"{message_id}-{part_counter}"
                    part_counter += 1

                    if isinstance(part, TextPart):
                        text_part = OpenCodeTextPart(
                            id=part_id,
                            session_id=session_id,
                            message_id=message_id,
                            text=part.content,
                            time=TimeStartEndOptional(start=now_ms),
                        )
                        part_file = parts_dir / f"{part_id}.json"
                        dct = text_part.model_dump(by_alias=True)
                        part_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

                    elif isinstance(part, ThinkingPart):
                        reasoning_part = OpenCodeReasoningPart(
                            id=part_id,
                            session_id=session_id,
                            message_id=message_id,
                            text=part.content,
                            time=TimeStartEndOptional(start=now_ms),
                        )
                        part_file = parts_dir / f"{part_id}.json"
                        dct = reasoning_part.model_dump(by_alias=True)
                        part_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

                    elif isinstance(part, ToolCallPart):
                        # Create tool part with pending status
                        tool_part = OpenCodeToolPart(
                            id=part_id,
                            session_id=session_id,
                            message_id=message_id,
                            call_id=part.tool_call_id,
                            tool=part.tool_name,
                            state=ToolStatePending(input=safe_args_as_dict(part)),
                        )
                        part_file = parts_dir / f"{part_id}.json"
                        dct = tool_part.model_dump(by_alias=True)
                        part_file.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages: list[ChatMessage[str]] = []
        # Narrow session list when a specific session is requested
        if query.name:
            sessions = [(sid, p) for sid, p in self._list_sessions() if sid == query.name]
        else:
            sessions = self._list_sessions()
        for session_id, session_path in sessions:
            if not session_path.exists():
                continue
            oc_messages = self._read_messages(session_id)
            # Read parts for all messages
            msg_parts_map = {oc_msg.id: self._read_parts(oc_msg.id) for oc_msg in oc_messages}
            for oc_msg in oc_messages:
                parts = msg_parts_map.get(oc_msg.id, [])
                chat_msg = helpers.to_chat_message(oc_msg, parts, session_id)

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
        session_id: str,
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
        """Log a message to OpenCode format."""
        from agentpool.storage.serialization import messages_adapter

        if not messages:
            logger.debug("No structured messages to log, skipping")
            return

        try:
            await self._write_message(
                message_id=message_id,
                session_id=session_id,
                role=role,
                model_messages=messages_adapter.validate_json(messages),
                parent_id=parent_id,
                model=model,
                cost_info=cost_info,
                finish_reason=finish_reason,
            )
        except Exception as e:
            logger.exception("Failed to write OpenCode message", error=str(e))

    async def log_session(
        self,
        *,
        session_id: str,
        node_name: str,
        start_time: datetime | None = None,
        model: str | None = None,
    ) -> None:
        """Log a conversation start."""
        # No-op for read-only provider

    async def get_sessions(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations with their messages."""
        result: list[tuple[ConvData, Sequence[ChatMessage[str]]]] = []
        for session_id, session_path in self._list_sessions():
            session = helpers.read_session(session_path)
            if not session:
                continue
            oc_messages = self._read_messages(session_id)
            if not oc_messages:
                continue
            # Read parts for all messages
            msg_parts_map = {oc_msg.id: self._read_parts(oc_msg.id) for oc_msg in oc_messages}
            # Convert messages
            chat_messages: list[ChatMessage[str]] = []
            total_tokens = 0
            total_cost = 0.0
            for oc_msg in oc_messages:
                parts = msg_parts_map.get(oc_msg.id, [])
                chat_msg = helpers.to_chat_message(oc_msg, parts, session_id)
                chat_messages.append(chat_msg)

                # Only assistant messages have tokens and cost
                if isinstance(oc_msg, AssistantMessage):
                    if oc_msg.tokens:
                        total_tokens += oc_msg.tokens.input + oc_msg.tokens.output
                    if oc_msg.cost:
                        total_cost += oc_msg.cost

            if not chat_messages:
                continue
            first_timestamp = ms_to_datetime(session.time.created)
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
                cost = chat_msg.cost_info
                msg_data: MessageData = {
                    "role": chat_msg.role,
                    "content": chat_msg.content,
                    "timestamp": (chat_msg.timestamp or get_now()).isoformat(),
                    "parent_id": chat_msg.parent_id,
                    "model": chat_msg.model_name,
                    "name": chat_msg.name,
                    "token_usage": TokenUsage(
                        total=cost.token_usage.total_tokens if cost else 0,
                        prompt=cost.token_usage.input_tokens if cost else 0,
                        completion=cost.token_usage.output_tokens if cost else 0,
                    )
                    if cost
                    else None,
                    "cost": float(cost.total_cost) if cost else None,
                    "response_time": chat_msg.response_time,
                }
                msg_data_list.append(msg_data)

            usage = TokenUsage(total=total_tokens, prompt=0, completion=0) if total_tokens else None
            conv_data = ConvData(
                id=session_id,
                agent=chat_messages[0].name or "opencode",
                title=session.title,
                start_time=first_timestamp.isoformat(),
                messages=msg_data_list,
                token_usage=usage,
            )
            result.append((conv_data, chat_messages))
            if filters.limit and len(result) >= filters.limit:
                break

        return result

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
        """Get conversation statistics."""
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_tokens": 0, "messages": 0, "models": set(), "total_cost": 0.0}
        )
        for _session_id, session_path in self._list_sessions():
            session = helpers.read_session(session_path)
            if not session:
                continue

            timestamp = ms_to_datetime(session.time.created)
            if timestamp < filters.cutoff:
                continue
            for oc_msg in self._read_messages(session.id):
                if not isinstance(oc_msg, AssistantMessage):
                    continue
                # AssistantMessage only has model_id
                model = oc_msg.model_id or "unknown"
                tokens = 0
                if oc_msg.tokens:
                    tokens = oc_msg.tokens.input + oc_msg.tokens.output

                msg_timestamp = ms_to_datetime(oc_msg.time.created)
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

    async def reset(self, *, agent_name: str | None = None, hard: bool = False) -> tuple[int, int]:
        """Reset storage.

        Warning: This would delete OpenCode data!
        """
        logger.warning("Reset not implemented for OpenCode storage (read-only)")
        return 0, 0

    async def get_session_counts(self, *, agent_name: str | None = None) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        conv_count = 0
        msg_count = 0
        for session_id, _session_path in self._list_sessions():
            msg_dir = self.messages_path / session_id
            if not msg_dir.exists():
                continue
            # Count JSON files directly instead of parsing them
            if message_files := list(msg_dir.glob("*.json")):
                conv_count += 1
                msg_count += len(message_files)

        return conv_count, msg_count

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a session."""
        # Read messages for this session
        messages: list[ChatMessage[str]] = []
        for oc_msg in self._read_messages(session_id):
            parts = self._read_parts(oc_msg.id)
            chat_msg = helpers.to_chat_message(oc_msg, parts, session_id)
            messages.append(chat_msg)

        # Sort by timestamp
        now = get_now()
        messages.sort(key=lambda m: m.timestamp or now)
        if not include_ancestors or not messages:
            return messages
        # Get ancestor chain if first message has parent_id
        if parent_id := messages[0].parent_id:
            ancestors = await self.get_message_ancestry(parent_id, session_id=session_id)
            return ancestors + messages
        return messages

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[str] | None:
        """Get a single message by ID."""
        # If session_id is provided, search only that session
        sessions = [(session_id, None)] if session_id else self._list_sessions()
        for sid, _session_path in sessions:
            assert sid is not None
            for oc_msg in self._read_messages(sid):
                if oc_msg.id == message_id:
                    parts = self._read_parts(oc_msg.id)
                    return helpers.to_chat_message(oc_msg, parts, sid)

        return None

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Traverses parent_id chain to build full history.
        When session_id is provided, loads messages once and traverses in-memory.

        Args:
            message_id: ID of the message
            session_id: Optional session ID hint for faster lookup

        Returns:
            List of messages from oldest ancestor to the specified message
        """
        # Fast path: if we know the session, load messages once and traverse in-memory
        ancestors: list[ChatMessage[str]] = []
        if session_id:
            # Build ID -> message index for O(1) lookups
            msg_by_id = {msg.id: msg for msg in self._read_messages(session_id)}
            current_id: str | None = message_id
            while current_id:
                oc_msg = msg_by_id.get(current_id)
                if not oc_msg:
                    break
                parts = self._read_parts(oc_msg.id)
                chat_msg = helpers.to_chat_message(oc_msg, parts, session_id)
                ancestors.append(chat_msg)
                current_id = chat_msg.parent_id
            ancestors.reverse()
            return ancestors
        # Slow path: search all sessions
        current_id = message_id
        while current_id:
            msg = await self.get_message(current_id)
            if not msg:
                break
            ancestors.append(msg)
            current_id = msg.parent_id
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
        """Fork a conversation at a specific point.

        Creates a new session directory. The fork point message_id is returned
        so callers can set it as parent_id for new messages.

        Args:
            source_session_id: Source session ID
            new_session_id: New session ID
            fork_from_message_id: Message ID to fork from. If None, forks from last
            new_agent_name: Not directly stored in OpenCode session format

        Returns:
            The ID of the fork point message
        """
        # Find source session
        source_path = next(
            (p for sid, p in self._list_sessions() if sid == source_session_id),
            None,
        )
        if not source_path:
            raise ValueError(f"Source conversation not found: {source_session_id}")
        source_session = helpers.read_session(source_path)
        assert source_session
        # Read source messages
        oc_messages = self._read_messages(source_session_id)
        # Find fork point
        fork_point_id: str | None = None
        if fork_from_message_id:
            # Verify message exists
            if not any(m.id == fork_from_message_id for m in oc_messages):
                raise ValueError(f"Message {fork_from_message_id} not found in conversation")
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
        # Create new session metadata
        fork_title = f"{source_session.title} (fork)" if source_session.title else "Forked Session"
        now = datetime_to_ms(get_now())
        new_session = Session(
            id=new_session_id,
            project_id=project_id,
            directory=source_session.directory,  # Same project directory as source
            title=fork_title,
            version=OPENCODE_VERSION,
            time=TimeCreatedUpdated(created=now, updated=now),
            summary=SessionSummary(files=0, additions=0, deletions=0),
        )
        # Write session file
        dct = new_session.model_dump(by_alias=True)
        new_session_path = new_session_dir / f"{new_session_id}.json"
        new_session_path.write_text(anyenv.dump_json(dct, indent=True), encoding="utf-8")
        # Create message and part directories
        (self.messages_path / new_session_id).mkdir(parents=True, exist_ok=True)
        (self.parts_path / new_session_id).mkdir(parents=True, exist_ok=True)
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
        conversations = await provider.get_sessions(filters)
        print(f"\nFound {len(conversations)} conversations")

        for conv_data, messages in conversations[:5]:
            print(f"  - {conv_data['id'][:8]}... | {conv_data['title'] or 'Untitled'}")
            print(f"    Messages: {len(messages)}, Updated: {conv_data['start_time']}")

        # Get counts
        conv_count, msg_count = await provider.get_session_counts()
        print(f"\nTotal: {conv_count} conversations, {msg_count} messages")

        # Get stats
        stats_filters = StatsFilters(
            cutoff=dt.datetime.now(dt.UTC) - dt.timedelta(days=30),
            group_by="day",
        )
        stats = await provider.get_session_stats(stats_filters)
        print(f"\nStats: {stats}")

    asyncio.run(main())
