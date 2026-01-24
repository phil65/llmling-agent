"""Claude storage converters."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
import uuid

import anyenv
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

from agentpool.messaging import ChatMessage, TokenCost
from agentpool.utils.time_utils import get_now, parse_iso_timestamp
from agentpool_storage.claude_provider.models import (
    ClaudeApiMessage,
    ClaudeAssistantEntry,
    ClaudeMessageContent,
    ClaudeUsage,
    ClaudeUserEntry,
    ClaudeUserMessage,
)


if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from agentpool_storage.claude_provider.models import ClaudeJSONLEntry


def chat_message_to_entry(
    message: ChatMessage[str],
    session_id: str,
    parent_uuid: str | None = None,
    cwd: str | None = None,
) -> ClaudeUserEntry | ClaudeAssistantEntry:
    """Convert a ChatMessage to a Claude JSONL entry."""
    msg_uuid = message.message_id or str(uuid.uuid4())
    timestamp = (message.timestamp or get_now()).isoformat().replace("+00:00", "Z")
    # Build entry based on role
    if message.role == "user":
        user_msg = ClaudeUserMessage(role="user", content=message.content)
        return ClaudeUserEntry(
            type="user",
            uuid=msg_uuid,
            parent_uuid=parent_uuid,
            session_id=session_id,
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
        session_id=session_id,
        timestamp=timestamp,
        message=assistant_msg,
        cwd=cwd or "",
        version="agentpool",
        user_type="external",
        is_sidechain=False,
    )


def extract_text_content(message: ClaudeApiMessage | ClaudeUserMessage) -> str:
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


def normalize_model_name(model: str | None) -> str | None:
    """Normalize Claude model names to simple IDs.

    Claude storage uses full model names like 'claude-opus-4-5-20251101'
    but Claude Code agent exposes simple IDs like 'opus', 'sonnet', 'haiku'.
    This normalizes to simple IDs for consistency with get_available_models().
    """
    if model is None:
        return None
    model_lower = model.lower()
    if "opus" in model_lower:
        return "opus"
    if "sonnet" in model_lower:
        return "sonnet"
    if "haiku" in model_lower:
        return "haiku"
    # Return original if not a known Claude model
    return model


def entry_to_chat_message(
    entry: ClaudeJSONLEntry,
    session_id: str,
    tool_id_mapping: dict[str, str] | None = None,
) -> ChatMessage[str] | None:
    """Convert a Claude JSONL entry to a ChatMessage.

    Reconstructs pydantic-ai ModelRequest/ModelResponse objects and stores
    them in the messages field for full fidelity.

    Args:
        entry: The JSONL entry to convert
        session_id: ID for the conversation
        tool_id_mapping: Optional mapping from tool_call_id to tool_name
            for resolving tool names in ToolReturnPart

    Returns None for non-message entries (queue-operation, summary, etc.).
    """
    # Only handle user/assistant entries with messages
    if not isinstance(entry, (ClaudeUserEntry, ClaudeAssistantEntry)):
        return None

    message = entry.message
    timestamp = parse_iso_timestamp(entry.timestamp)
    # Extract display content (text only for UI)
    content = extract_text_content(message)
    # Build pydantic-ai message
    pydantic_message = build_pydantic_message(entry, message, timestamp, tool_id_mapping or {})
    # Extract token usage and cost
    cost_info = None
    model = None
    finish_reason = None
    if isinstance(entry, ClaudeAssistantEntry) and isinstance(message, ClaudeApiMessage):
        usage = message.usage
        input_tokens = (
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        )
        output_tokens = usage.output_tokens
        if input_tokens or output_tokens:
            cost_info = TokenCost(
                token_usage=RunUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                total_cost=Decimal(0),  # Claude doesn't store cost directly
            )
        model = normalize_model_name(message.model)
        finish_reason = message.stop_reason

    return ChatMessage[str](
        content=content,
        session_id=session_id,
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


def build_pydantic_message(
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
                    tool_content = extract_tool_result_content(block)
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

    resp_parts: list[TextPart | ToolCallPart | ThinkingPart] = []
    usage = RequestUsage(
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cache_read_tokens=message.usage.cache_read_input_tokens,
        cache_write_tokens=message.usage.cache_creation_input_tokens,
    )

    if isinstance(msg_content, str):
        resp_parts.append(TextPart(content=msg_content))
    else:
        for block in msg_content:
            if block.type == "text" and block.text:
                resp_parts.append(TextPart(content=block.text))
            elif block.type == "thinking" and block.thinking:
                resp_parts.append(ThinkingPart(content=block.thinking, signature=block.signature))
            elif block.type == "tool_use" and block.id and block.name:
                args = block.input or {}
                resp_parts.append(
                    ToolCallPart(tool_name=block.name, args=args, tool_call_id=block.id)
                )

    if not resp_parts:
        return None
    model = normalize_model_name(message.model)
    return ModelResponse(parts=resp_parts, usage=usage, model_name=model, timestamp=timestamp)


def extract_tool_result_content(block: ClaudeMessageContent) -> str:
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


def encode_project_path(path: str) -> str:
    """Encode a project path to Claude's format.

    Claude encodes paths by replacing / with - and prepending -.
    Example: /home/user/project -> -home-user-project
    """
    return path.replace("/", "-")


def decode_project_path(encoded: str) -> str:
    """Decode a Claude project path back to filesystem path.

    Example: -home-user-project -> /home/user/project
    """
    encoded = encoded.removeprefix("-")
    return "/" + encoded.replace("-", "/")


def extract_title(session_path: Path, max_chars: int = 60) -> str | None:
    """Extract title from session file efficiently.

    Looks for a summary entry first, then falls back to the first line
    of the first user message. Stops reading as soon as a title is found.

    Args:
        session_path: Path to the JSONL session file
        max_chars: Maximum characters for title (truncates with '...')

    Returns:
        Extracted title or None if no suitable content found
    """
    if not session_path.exists():
        return None

    try:
        with session_path.open(encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                # Summary entries take priority
                if '"type":"summary"' in line:
                    try:
                        entry = anyenv.load_json(line, return_type=dict)
                        if summary := entry.get("summary"):
                            return str(summary)
                    except anyenv.JsonLoadError:
                        pass

                # First user message as fallback - stop here
                if '"type":"user"' in line:
                    try:
                        entry = anyenv.load_json(line, return_type=dict)
                        msg = entry.get("message", {})
                        content = msg.get("content", "")
                        if isinstance(content, str) and content:
                            # Use first line only, strip whitespace
                            first_line = content.split("\n")[0].strip()
                            if len(first_line) > max_chars:
                                return first_line[:max_chars] + "..."
                            return first_line if first_line else None
                    except anyenv.JsonLoadError:
                        pass
                    break  # Stop after first user message
    except OSError:
        pass

    return None
