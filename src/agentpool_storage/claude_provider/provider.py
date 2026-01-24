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

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anyenv
from pydantic import TypeAdapter

from agentpool.common_types import MessageRole
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.utils.time_utils import get_now, parse_iso_timestamp
from agentpool_storage.base import StorageProvider
from agentpool_storage.claude_provider.converters import (
    chat_message_to_entry,
    encode_project_path,
    entry_to_chat_message,
    extract_title,
    normalize_model_name,
)
from agentpool_storage.claude_provider.models import (
    ClaudeApiMessage,
    ClaudeAssistantEntry,
    ClaudeJSONLEntry,
    ClaudeUserEntry,
)
from agentpool_storage.models import TokenUsage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import FinishReason

    from agentpool.messaging import TokenCost
    from agentpool_config.session import SessionQuery
    from agentpool_config.storage import ClaudeStorageConfig
    from agentpool_storage.models import ConversationData, MessageData, QueryFilters, StatsFilters


logger = get_logger(__name__)


def write_entry(session_path: Path, entry: ClaudeJSONLEntry) -> None:
    """Append an entry to a session file."""
    session_path.parent.mkdir(parents=True, exist_ok=True)
    with session_path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json(by_alias=True) + "\n")


def _build_tool_id_mapping(entries: list[ClaudeJSONLEntry]) -> dict[str, str]:
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


def _read_session(session_path: Path) -> list[ClaudeJSONLEntry]:
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
                data = anyenv.load_json(stripped, return_type=dict)
                entry = adapter.validate_python(data)
                entries.append(entry)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Failed to parse JSONL line",
                    path=str(session_path),
                    error=str(e),
                    raw_line=raw_line,
                )
    return entries


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

    def _get_project_dir(self, project_path: str) -> Path:
        """Get the directory for a project's conversations."""
        return self.projects_path / encode_project_path(project_path)

    def _find_session_path(self, session_id: str) -> Path | None:
        """Find the file path for a session by ID.

        Searches across all project directories.

        Args:
            session_id: Session ID (file stem)

        Returns:
            Path to the session file, or None if not found
        """
        for project_dir in self.projects_path.iterdir():
            if project_dir.is_dir():
                candidate = project_dir / f"{session_id}.jsonl"
                if candidate.exists():
                    return candidate
        return None

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

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages: list[ChatMessage[str]] = []
        # Determine which sessions to search
        sessions = self._list_sessions()
        for session_id, session_path in sessions:
            # Filter by conversation/session name if specified
            if query.name and session_id != query.name:
                continue

            entries = _read_session(session_path)
            tool_mapping = _build_tool_id_mapping(entries)

            for entry in entries:
                msg = entry_to_chat_message(entry, session_id, tool_mapping)
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
        finish_reason: FinishReason | None = None,
    ) -> None:
        """Log a message to Claude format.

        Note: session_id should be in format "project_path:session_id"
        or just "session_id" (will use default project).
        """
        # Parse session_id
        if ":" in session_id:
            project_path, session_id = session_id.split(":", 1)
        else:
            project_path = "/tmp"

        # Build ChatMessage for conversion
        chat_message = ChatMessage[str](
            content=content,
            session_id=session_id,
            role=cast(MessageRole, role),
            message_id=message_id,
            name=name,
            model_name=model,
            cost_info=cost_info,
            response_time=response_time,
            parent_id=parent_id,
        )

        # Convert to entry and write
        entry = chat_message_to_entry(
            chat_message,
            session_id=session_id,
            parent_uuid=parent_id,
            cwd=project_path,
        )

        session_path = self._get_project_dir(project_path) / f"{session_id}.jsonl"
        write_entry(session_path, entry)

    async def log_session(
        self,
        *,
        session_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log a conversation start.

        In Claude format, conversations are implicit (created when first message is written).
        This is a no-op but could be extended to create an initial entry.
        """

    async def get_sessions(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations with their messages."""
        from agentpool_storage.models import ConversationData

        result: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []
        # Use cwd filter at filesystem level for efficiency
        sessions = self._list_sessions(project_path=filters.cwd)

        for session_id, session_path in sessions:
            entries = _read_session(session_path)
            if not entries:
                continue

            tool_mapping = _build_tool_id_mapping(entries)
            # Build messages
            messages: list[ChatMessage[str]] = []
            first_timestamp: datetime | None = None
            total_tokens = 0
            for entry in entries:
                msg = entry_to_chat_message(entry, session_id, tool_mapping)
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
            conv_data = ConversationData(
                id=session_id,
                agent=messages[0].name or "claude",
                title=extract_title(session_path),
                start_time=(first_timestamp or get_now()).isoformat(),
                messages=msg_data_list,
                token_usage=token_usage_data,
            )

            result.append((conv_data, messages))
            if filters.limit and len(result) >= filters.limit:
                break

        return result

    async def get_session_stats(self, filters: StatsFilters) -> dict[str, dict[str, Any]]:
        """Get conversation statistics."""
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_tokens": 0, "messages": 0, "models": set()}
        )
        for _session_id, session_path in self._list_sessions():
            entries = _read_session(session_path)

            for entry in entries:
                if not isinstance(entry, ClaudeAssistantEntry):
                    continue

                if not isinstance(entry.message, ClaudeApiMessage):
                    continue

                api_msg = entry.message
                model = normalize_model_name(api_msg.model)
                usage = api_msg.usage
                total_tokens = (
                    usage.input_tokens + usage.output_tokens + usage.cache_read_input_tokens
                )
                timestamp = parse_iso_timestamp(entry.timestamp)
                if timestamp < filters.cutoff:  # Apply time filter
                    continue
                # Group by specified criterion
                match filters.group_by:
                    case "model":
                        key = model or "unknown"
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

    async def reset(self, *, agent_name: str | None = None, hard: bool = False) -> tuple[int, int]:
        """Reset storage.

        Warning: This will delete Claude conversation files!
        """
        conv_count = 0
        msg_count = 0
        for _session_id, session_path in self._list_sessions():
            entries = _read_session(session_path)
            msg_count += len([
                e for e in entries if isinstance(e, (ClaudeUserEntry, ClaudeAssistantEntry))
            ])
            conv_count += 1

            if hard or not agent_name:
                session_path.unlink(missing_ok=True)

        return conv_count, msg_count

    async def get_session_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        conv_count = 0
        msg_count = 0
        for _session_id, session_path in self._list_sessions():
            entries = _read_session(session_path)
            msg_entries = [
                e for e in entries if isinstance(e, (ClaudeUserEntry, ClaudeAssistantEntry))
            ]

            if msg_entries:
                conv_count += 1
                msg_count += len(msg_entries)

        return conv_count, msg_count

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a session.

        Args:
            session_id: Session ID (conversation ID in Claude format)
            include_ancestors: If True, traverse parent_uuid chain to include
                messages from ancestor conversations

        Returns:
            List of messages ordered by timestamp
        """
        # Find the session file
        session_path = None
        for sid, spath in self._list_sessions():
            if sid == session_id:
                session_path = spath
                break

        if not session_path:
            return []

        # Read entries and convert to messages
        entries = _read_session(session_path)
        tool_mapping = _build_tool_id_mapping(entries)
        messages = [
            m for entry in entries if (m := entry_to_chat_message(entry, session_id, tool_mapping))
        ]
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp or get_now())
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
        """Get a single message by ID.

        Args:
            message_id: UUID of the message
            session_id: Optional session ID hint for faster lookup

        Returns:
            The message if found, None otherwise
        """
        # If session_id is provided, search only that session
        sessions = (
            [(session_id, p)]
            if session_id and (p := self._find_session_path(session_id))
            else self._list_sessions()
        )
        for sid, session_path in sessions:
            entries = _read_session(session_path)
            tool_mapping = _build_tool_id_mapping(entries)
            for entry in entries:
                if (
                    isinstance(entry, (ClaudeUserEntry, ClaudeAssistantEntry))
                    and entry.uuid == message_id
                ):
                    return entry_to_chat_message(entry, sid, tool_mapping)

        return None

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Traverses parent_uuid chain to build full history.
        When session_id is provided, loads the session once and traverses in-memory.

        Args:
            message_id: UUID of the message
            session_id: Optional session ID hint for faster lookup

        Returns:
            List of messages from oldest ancestor to the specified message
        """
        # Fast path: if we know the session, load once and traverse in-memory
        if session_id:
            session_path = self._find_session_path(session_id)
            if session_path:
                entries = _read_session(session_path)
                tool_mapping = _build_tool_id_mapping(entries)
                # Build UUID -> entry index for O(1) lookups
                entry_by_uuid: dict[str, ClaudeUserEntry | ClaudeAssistantEntry] = {}
                for entry in entries:
                    if isinstance(entry, (ClaudeUserEntry, ClaudeAssistantEntry)):
                        entry_by_uuid[entry.uuid] = entry

                ancestors: list[ChatMessage[str]] = []
                current_id: str | None = message_id
                while current_id:
                    found = entry_by_uuid.get(current_id)
                    if not found:
                        break
                    msg = entry_to_chat_message(found, session_id, tool_mapping)
                    if not msg:
                        break
                    ancestors.append(msg)
                    current_id = msg.parent_id
                ancestors.reverse()
                return ancestors

        # Slow path: search all sessions
        ancestors = []
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

        Creates a new session file. The fork point message_id is returned
        so callers can set it as parent_uuid for new messages.

        Args:
            source_session_id: Source session ID
            new_session_id: New session ID
            fork_from_message_id: UUID to fork from. If None, forks from last message
            new_agent_name: Not used in Claude format (no agent metadata in sessions)

        Returns:
            The UUID of the fork point message
        """
        # Find source session
        sessions = self._list_sessions()
        source_path = next((spath for sid, spath in sessions if sid == source_session_id), None)
        if not source_path:
            raise ValueError(f"Source conversation not found: {source_session_id}")
        # Read source entries
        entries = _read_session(source_path)
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
            msg_entries = [
                e for e in entries if isinstance(e, (ClaudeUserEntry, ClaudeAssistantEntry))
            ]
            if msg_entries:
                fork_point_id = msg_entries[-1].uuid

        # Create new session file (empty for now - will be populated when messages added)
        # Determine project from source path structure
        project_name = source_path.parent.name
        new_path = self.projects_path / project_name / f"{new_session_id}.jsonl"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.touch()
        return fork_point_id


if __name__ == "__main__":
    from agentpool_config.storage import ClaudeStorageConfig

    cfg = ClaudeStorageConfig()
    provider = ClaudeStorageProvider(cfg)
    print(provider._list_sessions())
