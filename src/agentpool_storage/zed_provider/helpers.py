"""Helper functions for Zed storage provider.

Stateless conversion and utility functions for working with Zed format.
"""

from __future__ import annotations

import base64
from datetime import datetime
import io
import json
from typing import Any

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
from agentpool.utils.now import get_now
from agentpool_storage.zed_provider.models import ZedThread


logger = get_logger(__name__)


def decompress_thread(data: bytes, data_type: str) -> ZedThread:
    """Decompress and parse thread data.

    Args:
        data: Compressed thread data
        data_type: Type of compression ("zstd" or plain)

    Returns:
        Parsed ZedThread object
    """
    if data_type == "zstd":
        dctx = zstandard.ZstdDecompressor()
        # Use stream_reader for data without content size in header
        reader = dctx.stream_reader(io.BytesIO(data))
        json_data = reader.read()
    else:
        json_data = data

    thread_dict = json.loads(json_data)
    return ZedThread.model_validate(thread_dict)


def parse_user_content(  # noqa: PLR0915
    content_list: list[dict[str, Any]],
) -> tuple[str, list[str | BinaryContent]]:
    """Parse user message content blocks.

    Args:
        content_list: List of content blocks from Zed user message

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


def parse_tool_results(tool_results: dict[str, Any]) -> list[ToolReturnPart]:
    """Parse tool results into ToolReturnParts.

    Args:
        tool_results: Dictionary of tool results from Zed agent message

    Returns:
        List of ToolReturnPart objects
    """
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


def thread_to_chat_messages(thread: ZedThread, thread_id: str) -> list[ChatMessage[str]]:
    """Convert a Zed thread to ChatMessages.

    Args:
        thread: Zed thread object
        thread_id: Thread identifier

    Returns:
        List of ChatMessage objects
    """
    messages: list[ChatMessage[str]] = []
    try:
        updated_at = datetime.fromisoformat(thread.updated_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        updated_at = get_now()
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
            display_text, pydantic_content = parse_user_content(user_msg.content)
            part = UserPromptPart(content=pydantic_content)
            model_request = ModelRequest(parts=[part])

            messages.append(
                ChatMessage[str](
                    content=display_text,
                    conversation_id=thread_id,
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
            display_text, pydantic_parts = parse_agent_content(agent_msg.content)
            # Build ModelResponse
            usage = RequestUsage()
            model_response = ModelResponse(parts=pydantic_parts, usage=usage, model_name=model_name)
            # Build tool return parts for next request if there are tool results
            tool_return_parts = parse_tool_results(agent_msg.tool_results)
            # Create the messages list - response, then optionally tool returns
            pydantic_messages: list[ModelResponse | ModelRequest] = [model_response]
            if tool_return_parts:
                pydantic_messages.append(ModelRequest(parts=tool_return_parts))

            messages.append(
                ChatMessage[str](
                    content=display_text,
                    conversation_id=thread_id,
                    role="assistant",
                    message_id=msg_id,
                    name="zed",
                    model_name=model_name,
                    timestamp=updated_at,
                    messages=pydantic_messages,
                )
            )

    return messages
