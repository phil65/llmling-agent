"""Helper functions for Zed storage provider.

Stateless conversion and utility functions for working with Zed format.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any, Literal

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
import zstandard

from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.utils.time_utils import parse_iso_timestamp


if TYPE_CHECKING:
    from datetime import datetime

    from agentpool_storage.zed_provider.models import (
        ZedFlatMessage,
        ZedNestedMessage,
        ZedThread,
        ZedToolResult,
    )


logger = get_logger(__name__)


# Module-level reusable decompressor (stateless, thread-safe for decompression)
_ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()


def _decompress(data: bytes, data_type: Literal["zstd", "plain"]) -> bytes:
    """Decompress thread data.

    Args:
        data: Compressed thread data
        data_type: Type of compression ("zstd" or plain)

    Returns:
        Decompressed bytes
    """
    if data_type == "zstd":
        reader = _ZSTD_DECOMPRESSOR.stream_reader(io.BytesIO(data))
        return reader.read()
    return data


def detect_image_media_type(data: bytes) -> str:
    """Detect image media type from magic bytes.

    Args:
        data: Raw image bytes (at least first 12 bytes needed)

    Returns:
        Media type string (defaults to "image/png" if unknown)
    """
    match data[:12]:
        case b if b[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        case b if b[:4] == b"\x89PNG":
            return "image/png"
        case b if b[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        case b if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
            return "image/webp"
        case _:
            return "image/png"


def parse_user_content(items: list[dict[str, Any]]) -> tuple[str, list[str | BinaryContent]]:
    """Parse user message content blocks.

    Args:
        items: List of content blocks from Zed user message

    Returns:
        Tuple of (display_text, pydantic_ai_content_list)
    """
    display_parts: list[str] = []
    pydantic_content: list[str | BinaryContent] = []

    for item in items:
        match item:
            case {"Text": text}:
                display_parts.append(text)
                pydantic_content.append(text)

            case {"Image": {"source": source}}:
                binary_data = base64.b64decode(source)
                media_type = detect_image_media_type(binary_data)
                pydantic_content.append(BinaryContent(data=binary_data, media_type=media_type))
                display_parts.append("[image]")
            case {"Mention": {"uri": uri, "content": content}}:
                match uri:
                    case {"File": {"abs_path": path}}:
                        formatted = f"[File: {path}]\n{content}"
                    case {"Directory": {"abs_path": path}}:
                        formatted = f"[Directory: {path}]\n{content}"
                    case {"Symbol": {"abs_path": path, "name": name}}:
                        formatted = f"[Symbol: {name} in {path}]\n{content}"
                    case {"Selection": {"abs_path": path}}:
                        formatted = f"[Selection: {path}]\n{content}"
                    case {"Fetch": {"url": url}}:
                        formatted = f"[Fetched: {url}]\n{content}"
                    case _:
                        formatted = content
                display_parts.append(formatted)
                pydantic_content.append(formatted)

    display_text = "\n".join(display_parts)
    return display_text, pydantic_content


def parse_agent_content(
    content_list: list[dict[str, Any]],
) -> tuple[str, list[TextPart | ThinkingPart | ToolCallPart]]:
    """Parse agent message content blocks.

    Args:
        content_list: List of content blocks from Zed agent message

    Returns:
        Tuple of (display_text, pydantic_ai_parts)
    """
    display_parts: list[str] = []
    pydantic_parts: list[TextPart | ThinkingPart | ToolCallPart] = []

    for item in content_list:
        match item:
            case {"Text": text}:
                display_parts.append(text)
                pydantic_parts.append(TextPart(content=text))
            case {"Thinking": {"text": text, **rest}}:
                signature = rest.get("signature")
                assert signature is None or isinstance(signature, str)
                display_parts.append(f"<thinking>\n{text}\n</thinking>")
                pydantic_parts.append(ThinkingPart(content=text, signature=signature))
            case {"ToolUse": tool_use}:
                tool_id = tool_use.get("id", "")
                tool_name = tool_use.get("name", "")
                tool_input = tool_use.get("input", {})
                display_parts.append(f"[Tool: {tool_name}]")
                part = ToolCallPart(tool_name=tool_name, args=tool_input, tool_call_id=tool_id)
                pydantic_parts.append(part)

    display_text = "\n".join(display_parts)
    return display_text, pydantic_parts


def parse_tool_results(tool_results: dict[str, ZedToolResult]) -> list[ToolReturnPart]:
    """Parse tool results into ToolReturnParts.

    Args:
        tool_results: Dictionary of tool results from Zed agent message

    Returns:
        List of ToolReturnPart objects
    """
    parts: list[ToolReturnPart] = []

    for tool_id, result in tool_results.items():
        match result.output, result.content:
            case {"Text": text}, _:
                output_str = text
            case _, {"Text": text}:
                output_str = text
            case _, str(text):
                output_str = text
            case str(text), _:
                output_str = text
            case _:
                output_str = ""

        parts.append(
            ToolReturnPart(tool_name=result.tool_name, content=output_str, tool_call_id=tool_id)
        )

    return parts


def _convert_flat_message(
    msg: ZedFlatMessage,
    thread_id: str,
    model_name: str | None,
    updated_at: datetime,
) -> ChatMessage[str]:
    """Convert a v0.1.0 flat message to ChatMessage."""
    msg_id = f"{thread_id}_{msg.id}"
    # Extract text from segments
    text_parts = [seg.text for seg in msg.segments if seg.text]
    display_text = "\n".join(text_parts)

    if msg.role == "user":
        part = UserPromptPart(content=display_text)
        return ChatMessage[str](
            content=display_text,
            session_id=thread_id,
            role="user",
            message_id=msg_id,
            timestamp=updated_at,
            messages=[ModelRequest(parts=[part])],
        )
    # assistant
    pydantic_parts: list[TextPart | ThinkingPart | ToolCallPart] = [TextPart(content=display_text)]
    model_response = ModelResponse(parts=pydantic_parts, model_name=model_name)
    return ChatMessage[str](
        content=display_text,
        session_id=thread_id,
        role="assistant",
        message_id=msg_id,
        name="zed",
        model_name=model_name,
        timestamp=updated_at,
        messages=[model_response],
    )


def _convert_nested_message(
    msg: ZedNestedMessage,
    thread_id: str,
    idx: int,
    model_name: str | None,
    updated_at: datetime,
) -> ChatMessage[str]:
    """Convert a v0.2.0+ nested message to ChatMessage."""
    msg_id = f"{thread_id}_{idx}"

    if msg.User is not None:
        user_msg = msg.User
        display_text, pydantic_content = parse_user_content(user_msg.content)
        part = UserPromptPart(content=pydantic_content)
        return ChatMessage[str](
            content=display_text,
            session_id=thread_id,
            role="user",
            message_id=user_msg.id or msg_id,
            timestamp=updated_at,
            messages=[ModelRequest(parts=[part])],
        )

    if msg.Agent is not None:
        agent_msg = msg.Agent
        display_text, pydantic_parts = parse_agent_content(agent_msg.content)
        model_response = ModelResponse(parts=pydantic_parts, model_name=model_name)
        pydantic_messages: list[ModelResponse | ModelRequest] = [model_response]
        if tool_return_parts := parse_tool_results(agent_msg.tool_results):
            pydantic_messages.append(ModelRequest(parts=tool_return_parts))
        return ChatMessage[str](
            content=display_text,
            session_id=thread_id,
            role="assistant",
            message_id=msg_id,
            name="zed",
            model_name=model_name,
            timestamp=updated_at,
            messages=pydantic_messages,
        )

    raise ValueError("Unexpected message type")


def thread_to_chat_messages(thread: ZedThread, thread_id: str) -> list[ChatMessage[str]]:
    """Convert a Zed thread to ChatMessages.

    Handles both v0.1.0 flat format and v0.2.0+ nested format.

    Args:
        thread: Zed thread object
        thread_id: Thread identifier

    Returns:
        List of ChatMessage objects
    """
    from agentpool_storage.zed_provider.models import ZedFlatMessage, ZedNestedMessage

    messages: list[ChatMessage[str]] = []
    updated_at = parse_iso_timestamp(thread.updated_at)
    model_name = f"{thread.model.provider}:{thread.model.model}" if thread.model else None

    for idx, msg in enumerate(thread.messages):
        match msg:
            case "Resume":
                continue  # Skip control messages
            case ZedFlatMessage():
                chat_msg = _convert_flat_message(msg, thread_id, model_name, updated_at)
                messages.append(chat_msg)
            case ZedNestedMessage():
                if chat_msg := _convert_nested_message(msg, thread_id, idx, model_name, updated_at):
                    messages.append(chat_msg)

    return messages
