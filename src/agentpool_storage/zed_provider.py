"""Zed IDE storage provider - reads from ~/.local/share/zed/threads format."""

from __future__ import annotations

import base64
from collections import defaultdict
from datetime import datetime
import io
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any, Literal

import anyenv
from pydantic import AliasChoices, BaseModel, Field
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
import zstandard

from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.utils.time_utils import get_now, parse_iso_timestamp
from agentpool_storage.base import StorageProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import FinishReason

    from agentpool.messaging import TokenCost
    from agentpool_config.session import SessionQuery
    from agentpool_config.storage import ZedStorageConfig
    from agentpool_storage.models import (
        ConversationData,
        MessageData,
        QueryFilters,
        StatsFilters,
        TokenUsage,
    )

logger = get_logger(__name__)


# Zed data models


class ZedMentionUri(BaseModel):
    """Mention URI - can be File, Directory, Symbol, etc."""

    File: dict[str, Any] | None = None
    Directory: dict[str, Any] | None = None
    Symbol: dict[str, Any] | None = None
    Selection: dict[str, Any] | None = None
    Thread: dict[str, Any] | None = None
    TextThread: dict[str, Any] | None = None
    Rule: dict[str, Any] | None = None
    Fetch: dict[str, Any] | None = None
    PastedImage: bool | None = None


class ZedMention(BaseModel):
    """A file/symbol mention in Zed."""

    uri: ZedMentionUri
    content: str


class ZedImage(BaseModel):
    """An image in Zed (base64 encoded)."""

    source: str  # base64 encoded


class ZedThinking(BaseModel):
    """Thinking block from model."""

    text: str
    signature: str | None = None


class ZedToolUse(BaseModel):
    """Tool use block."""

    id: str
    name: str
    raw_input: str
    input: dict[str, Any]
    is_input_complete: bool = True
    thought_signature: str | None = None


class ZedToolResult(BaseModel):
    """Tool result."""

    tool_use_id: str
    tool_name: str
    is_error: bool = False
    content: dict[str, Any] | str | None = None
    output: dict[str, Any] | str | None = None


class ZedUserMessage(BaseModel):
    """User message in Zed thread."""

    id: str
    content: list[dict[str, Any]]  # Can contain Text, Image, Mention


class ZedAgentMessage(BaseModel):
    """Agent message in Zed thread."""

    content: list[dict[str, Any]]  # Can contain Text, Thinking, ToolUse
    tool_results: dict[str, ZedToolResult] = Field(default_factory=dict)
    reasoning_details: Any | None = None


class ZedMessage(BaseModel):
    """A message in Zed thread - either User or Agent."""

    User: ZedUserMessage | None = None
    Agent: ZedAgentMessage | None = None


class ZedLanguageModel(BaseModel):
    """Model configuration."""

    provider: str
    model: str


class ZedWorktreeSnapshot(BaseModel):
    """Git worktree snapshot."""

    worktree_path: str
    git_state: dict[str, Any] | None = None


class ZedProjectSnapshot(BaseModel):
    """Project snapshot with git state."""

    worktree_snapshots: list[ZedWorktreeSnapshot] = Field(default_factory=list)
    unsaved_buffer_paths: list[str] = Field(default_factory=list)
    timestamp: str | None = None


class ZedThread(BaseModel):
    """A Zed conversation thread."""

    model_config = {"populate_by_name": True}

    # v0.3.0 uses "title", v0.2.0 uses "summary"
    title: str = Field(alias="title", validation_alias=AliasChoices("title", "summary"))
    messages: list[ZedMessage | Literal["Resume"]]  # Control messages
    updated_at: str
    version: str | None = None
    detailed_summary: str | None = None  # v0.3.0 field
    detailed_summary_state: str | dict[str, Any] | None = None  # v0.2.0 field
    initial_project_snapshot: ZedProjectSnapshot | None = None
    cumulative_token_usage: dict[str, int] = Field(default_factory=dict)
    request_token_usage: list[dict[str, int]] | dict[str, dict[str, int]] = Field(
        default_factory=list
    )  # Can be list or dict depending on version
    model: ZedLanguageModel | None = None
    completion_mode: str | None = None
    profile: str | None = None
    exceeded_window_error: Any | None = None
    tool_use_limit_reached: bool = False


def _decompress_thread(data: bytes, data_type: str) -> ZedThread:
    """Decompress and parse thread data."""
    if data_type == "zstd":
        dctx = zstandard.ZstdDecompressor()
        # Use stream_reader for data without content size in header
        reader = dctx.stream_reader(io.BytesIO(data))
        json_data = reader.read()
    else:
        json_data = data

    thread_dict = anyenv.load_json(json_data, return_type=dict)
    return ZedThread.model_validate(thread_dict)


def _parse_user_content(  # noqa: PLR0915
    content_list: list[dict[str, Any]],
) -> tuple[str, list[str | BinaryContent]]:
    """Parse user message content blocks.

    Returns:
        Tuple of (display_text, pydantic_ai_content_list)
    """
    display_parts: list[str] = []
    pydantic_content: list[str | BinaryContent] = []

    for item in content_list:
        if "Text" in item:
            text = item["Text"]
            display_parts.append(text)
            pydantic_content.append(text)

        elif "Image" in item:
            image_data = item["Image"]
            source = image_data.get("source", "")
            try:
                binary_data = base64.b64decode(source)
                # Try to detect image type from magic bytes
                media_type = "image/png"  # default
                if binary_data[:3] == b"\xff\xd8\xff":
                    media_type = "image/jpeg"
                elif binary_data[:4] == b"\x89PNG":
                    media_type = "image/png"
                elif binary_data[:6] in (b"GIF87a", b"GIF89a"):
                    media_type = "image/gif"
                elif binary_data[:4] == b"RIFF" and binary_data[8:12] == b"WEBP":
                    media_type = "image/webp"

                pydantic_content.append(BinaryContent(data=binary_data, media_type=media_type))
                display_parts.append("[image]")
            except (ValueError, TypeError, IndexError) as e:
                logger.warning("Failed to decode image", error=str(e))
                display_parts.append("[image decode error]")

        elif "Mention" in item:
            mention = item["Mention"]
            uri = mention.get("uri", {})
            content = mention.get("content", "")

            # Format mention based on type
            if "File" in uri:
                file_info = uri["File"]
                path = file_info.get("abs_path", "unknown")
                formatted = f"[File: {path}]\n{content}"
            elif "Directory" in uri:
                dir_info = uri["Directory"]
                path = dir_info.get("abs_path", "unknown")
                formatted = f"[Directory: {path}]\n{content}"
            elif "Symbol" in uri:
                symbol_info = uri["Symbol"]
                path = symbol_info.get("abs_path", "unknown")
                name = symbol_info.get("name", "")
                formatted = f"[Symbol: {name} in {path}]\n{content}"
            elif "Selection" in uri:
                sel_info = uri["Selection"]
                path = sel_info.get("abs_path", "unknown")
                formatted = f"[Selection: {path}]\n{content}"
            elif "Fetch" in uri:
                fetch_info = uri["Fetch"]
                url = fetch_info.get("url", "unknown")
                formatted = f"[Fetched: {url}]\n{content}"
            else:
                formatted = content

            display_parts.append(formatted)
            pydantic_content.append(formatted)

    display_text = "\n".join(display_parts)
    return display_text, pydantic_content


def _parse_agent_content(
    content_list: list[dict[str, Any]],
) -> tuple[str, list[TextPart | ThinkingPart | ToolCallPart]]:
    """Parse agent message content blocks.

    Returns:
        Tuple of (display_text, pydantic_ai_parts)
    """
    display_parts: list[str] = []
    pydantic_parts: list[TextPart | ThinkingPart | ToolCallPart] = []

    for item in content_list:
        if "Text" in item:
            text = item["Text"]
            display_parts.append(text)
            pydantic_parts.append(TextPart(content=text))

        elif "Thinking" in item:
            thinking = item["Thinking"]
            text = thinking.get("text", "")
            signature = thinking.get("signature")
            display_parts.append(f"<thinking>\n{text}\n</thinking>")
            pydantic_parts.append(ThinkingPart(content=text, signature=signature))

        elif "ToolUse" in item:
            tool_use = item["ToolUse"]
            tool_id = tool_use.get("id", "")
            tool_name = tool_use.get("name", "")
            tool_input = tool_use.get("input", {})
            display_parts.append(f"[Tool: {tool_name}]")
            pydantic_parts.append(
                ToolCallPart(tool_name=tool_name, args=tool_input, tool_call_id=tool_id)
            )

    display_text = "\n".join(display_parts)
    return display_text, pydantic_parts


def _parse_tool_results(tool_results: dict[str, Any]) -> list[ToolReturnPart]:
    """Parse tool results into ToolReturnParts."""
    parts: list[ToolReturnPart] = []

    for tool_id, result in tool_results.items():
        if isinstance(result, dict):
            tool_name = result.get("tool_name", "")
            output = result.get("output", "")
            content = result.get("content", {})

            # Handle output being a dict like {"Text": "..."}
            if isinstance(output, dict) and "Text" in output:
                output = output["Text"]
            # Extract text content if available
            elif isinstance(content, dict) and "Text" in content:
                output = content["Text"]
            elif isinstance(content, str):
                output = content

            parts.append(
                ToolReturnPart(tool_name=tool_name, content=output or "", tool_call_id=tool_id)
            )

    return parts


def _thread_to_chat_messages(thread: ZedThread, thread_id: str) -> list[ChatMessage[str]]:
    """Convert a Zed thread to ChatMessages."""
    messages: list[ChatMessage[str]] = []
    updated_at = parse_iso_timestamp(thread.updated_at)
    # Get model info
    model_name = None
    if thread.model:
        model_name = f"{thread.model.provider}:{thread.model.model}"

    for idx, msg in enumerate(thread.messages):
        if msg == "Resume":
            continue  # Skip control messages
        msg_id = f"{thread_id}_{idx}"

        if msg.User is not None:
            user_msg = msg.User
            display_text, pydantic_content = _parse_user_content(user_msg.content)
            part = UserPromptPart(content=pydantic_content)
            model_request = ModelRequest(parts=[part])

            messages.append(
                ChatMessage[str](
                    content=display_text,
                    session_id=thread_id,
                    role="user",
                    message_id=user_msg.id or msg_id,
                    name=None,
                    model_name=None,
                    timestamp=updated_at,  # Zed doesn't store per-message timestamps
                    messages=[model_request],
                )
            )

        elif msg.Agent is not None:
            agent_msg = msg.Agent
            display_text, pydantic_parts = _parse_agent_content(agent_msg.content)
            # Build ModelResponse
            usage = RequestUsage()
            model_response = ModelResponse(parts=pydantic_parts, usage=usage, model_name=model_name)
            # Build tool return parts for next request if there are tool results
            tool_return_parts = _parse_tool_results(agent_msg.tool_results)
            # Create the messages list - response, then optionally tool returns
            pydantic_messages: list[ModelResponse | ModelRequest] = [model_response]
            if tool_return_parts:
                pydantic_messages.append(ModelRequest(parts=tool_return_parts))

            messages.append(
                ChatMessage[str](
                    content=display_text,
                    session_id=thread_id,
                    role="assistant",
                    message_id=msg_id,
                    name="zed",
                    model_name=model_name,
                    timestamp=updated_at,
                    messages=pydantic_messages,
                )
            )

    return messages


class ZedStorageProvider(StorageProvider):
    """Storage provider that reads Zed IDE's native thread format.

    Zed stores conversations as zstd-compressed JSON in:
    - ~/.local/share/zed/threads/threads.db (SQLite)

    This is a READ-ONLY provider - it cannot write back to Zed's format.

    ## Supported Zed content types:
    - Text → str / TextPart
    - Image → BinaryContent
    - Mention (File, Directory, Symbol, Selection) → formatted str
    - Thinking → ThinkingPart
    - ToolUse → ToolCallPart
    - tool_results → ToolReturnPart

    ## Fields NOT available in Zed format:
    - Per-message token costs (only cumulative per thread)
    - Response time
    - Parent message ID (flat structure)
    - Forwarded from chain
    - Finish reason
    """

    can_load_history = True

    def __init__(self, config: ZedStorageConfig) -> None:
        """Initialize Zed storage provider.

        Args:
            config: Configuration for the provider
        """
        super().__init__(config)
        self.db_path = Path(config.path).expanduser()
        if not self.db_path.name.endswith(".db"):
            # If path is directory, add default db location
            self.db_path = self.db_path / "threads" / "threads.db"

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection."""
        if not self.db_path.exists():
            msg = f"Zed threads database not found: {self.db_path}"
            raise FileNotFoundError(msg)
        return sqlite3.connect(self.db_path)

    def _list_threads(self) -> list[tuple[str, str, str]]:
        """List all threads.

        Returns:
            List of (id, summary, updated_at) tuples
        """
        try:
            conn = self._get_connection()
            query = "SELECT id, summary, updated_at FROM threads ORDER BY updated_at DESC"
            cursor = conn.execute(query)
            threads = cursor.fetchall()
            conn.close()
        except FileNotFoundError:
            return []
        except sqlite3.Error as e:
            logger.warning("Failed to list Zed threads", error=str(e))
            return []
        else:
            return threads

    def _load_thread(self, thread_id: str) -> ZedThread | None:
        """Load a single thread by ID."""
        try:
            conn = self._get_connection()
            query = "SELECT data_type, data FROM threads WHERE id = ? LIMIT 1"
            cursor = conn.execute(query, (thread_id,))
            row = cursor.fetchone()
            conn.close()
            if row is None:
                return None

            data_type, data = row
            return _decompress_thread(data, data_type)
        except FileNotFoundError:
            return None
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.warning("Failed to load Zed thread", thread_id=thread_id, error=str(e))
            return None

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages based on query."""
        messages: list[ChatMessage[str]] = []
        # Narrow thread list when a specific name is requested
        if query.name:
            threads = [(tid, s, u) for tid, s, u in self._list_threads() if query.name in (tid, s)]
        else:
            threads = self._list_threads()
        for thread_id, _summary, _updated_at in threads:
            thread = self._load_thread(thread_id)
            if thread is None:
                continue
            for msg in _thread_to_chat_messages(thread, thread_id):
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
        """Log a message - NOT SUPPORTED (read-only provider)."""
        logger.warning("ZedStorageProvider is read-only, cannot log messages")

    async def log_session(
        self,
        *,
        session_id: str,
        node_name: str,
        start_time: datetime | None = None,
        model: str | None = None,
    ) -> None:
        """Log a conversation - NOT SUPPORTED (read-only provider)."""
        logger.warning("ZedStorageProvider is read-only, cannot log conversations")

    async def get_sessions(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations with their messages."""
        from agentpool_storage.models import ConversationData as ConvData

        result: list[tuple[ConvData, Sequence[ChatMessage[str]]]] = []
        for thread_id, summary, updated_at_str in self._list_threads():
            thread = self._load_thread(thread_id)
            if thread is None:
                continue
            # Parse timestamp
            updated_at = parse_iso_timestamp(updated_at_str)
            # Apply filters
            if filters.since and updated_at < filters.since:
                continue
            messages = _thread_to_chat_messages(thread, thread_id)
            if not messages:
                continue
            if filters.agent_name and not any(m.name == filters.agent_name for m in messages):
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
                    "token_usage": None,
                    "cost": None,
                    "response_time": None,
                }
                msg_data_list.append(msg_data)

            # Get token usage from thread-level cumulative data
            total_tokens = sum(thread.cumulative_token_usage.values())
            token_usage_data: TokenUsage | None = (
                {"total": total_tokens, "prompt": 0, "completion": 0} if total_tokens else None
            )

            conv_data = ConvData(
                id=thread_id,
                agent="zed",
                title=summary or thread.title,
                start_time=updated_at.isoformat(),
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
        for thread_id, _summary, updated_at_str in self._list_threads():
            timestamp = parse_iso_timestamp(updated_at_str)
            # Apply time filter
            if timestamp < filters.cutoff:
                continue
            thread = self._load_thread(thread_id)
            if thread is None:
                continue
            model_name = "unknown"
            if thread.model:
                model_name = f"{thread.model.provider}:{thread.model.model}"
            total_tokens = sum(thread.cumulative_token_usage.values())
            message_count = len(thread.messages)
            # Group by specified criterion
            match filters.group_by:
                case "model":
                    key = model_name
                case "hour":
                    key = timestamp.strftime("%Y-%m-%d %H:00")
                case "day":
                    key = timestamp.strftime("%Y-%m-%d")
                case _:
                    key = "zed"  # Default agent grouping

            stats[key]["messages"] += message_count
            stats[key]["total_tokens"] += total_tokens
            stats[key]["models"].add(model_name)

        # Convert sets to lists for JSON serialization
        for value in stats.values():
            value["models"] = list(value["models"])

        return dict(stats)

    async def reset(self, *, agent_name: str | None = None, hard: bool = False) -> tuple[int, int]:
        """Reset storage - NOT SUPPORTED (read-only provider)."""
        logger.warning("ZedStorageProvider is read-only, cannot reset")
        return 0, 0

    async def get_session_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts of conversations and messages."""
        conv_count = 0
        msg_count = 0
        for thread_id, _summary, _updated_at in self._list_threads():
            thread = self._load_thread(thread_id)
            if thread is None:
                continue

            conv_count += 1
            msg_count += len(thread.messages)
        return conv_count, msg_count

    async def get_session_title(self, session_id: str) -> str | None:
        """Get the title of a conversation."""
        thread = self._load_thread(session_id)
        if thread is None:
            return None
        return thread.title

    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[str]]:
        """Get all messages for a session.

        Args:
            session_id: Thread ID (conversation ID in Zed format)
            include_ancestors: If True, traverse parent_id chain to include
                messages from ancestor conversations (not supported in Zed format)

        Returns:
            List of messages ordered by timestamp

        Note:
            Zed threads don't have parent_id chain, so include_ancestors has no effect.
        """
        thread = self._load_thread(session_id)
        if thread is None:
            return []

        messages = _thread_to_chat_messages(thread, session_id)
        # Sort by timestamp (though they should already be in order)
        now = get_now()
        messages.sort(key=lambda m: m.timestamp or now)
        return messages

    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[str] | None:
        """Get a single message by ID.

        Args:
            message_id: ID of the message
            session_id: Optional session ID (thread_id) hint for faster lookup

        Returns:
            The message if found, None otherwise

        Note:
            Zed doesn't store individual message IDs, so this searches threads.
        """
        threads = [(session_id, None, None)] if session_id else self._list_threads()
        for thread_id, _summary, _updated_at in threads:
            thread = self._load_thread(thread_id)
            if thread is None:
                continue
            messages = _thread_to_chat_messages(thread, thread_id)
            if match := next((msg for msg in messages if msg.message_id == message_id), None):
                return match
        return None

    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get the ancestry chain of a message.

        Args:
            message_id: ID of the message
            session_id: Optional session ID (thread_id) hint for faster lookup

        Returns:
            List of messages from oldest ancestor to the specified message

        Note:
            Zed threads don't support parent_id chains, so this only returns
            the single message if found.
        """
        msg = await self.get_message(message_id, session_id=session_id)
        return [msg] if msg else []

    async def fork_conversation(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        """Fork a conversation at a specific point.

        Args:
            source_session_id: Source thread ID
            new_session_id: New thread ID
            fork_from_message_id: Message ID to fork from (not used - Zed is read-only)
            new_agent_name: Not used in Zed format

        Returns:
            None, as Zed storage is read-only

        Note:
            This is a READ-ONLY provider. Forking creates no persistent state.
            Returns None to indicate no fork point is available.
        """
        msg = "Fork conversation not supported for Zed storage (read-only)"
        logger.warning(msg, source=source_session_id, new=new_session_id)
        return None


if __name__ == "__main__":
    import asyncio
    import datetime as dt

    from agentpool_config.storage import ZedStorageConfig
    from agentpool_storage.models import QueryFilters, StatsFilters

    async def main() -> None:
        config = ZedStorageConfig()
        provider = ZedStorageProvider(config)
        print(f"Database: {provider.db_path}")
        print(f"Exists: {provider.db_path.exists()}")
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
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=30)
        stats_filters = StatsFilters(cutoff=cutoff, group_by="day")
        stats = await provider.get_session_stats(stats_filters)
        print(f"\nStats: {stats}")

    asyncio.run(main())
