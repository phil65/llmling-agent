"""Helper functions for OpenCode storage provider.

Stateless conversion and utility functions for working with OpenCode format.
"""

from __future__ import annotations

import base64
from decimal import Decimal
from typing import TYPE_CHECKING

import anyenv
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    RunUsage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)

from agentpool.log import get_logger
from agentpool.messaging import ChatMessage, TokenCost
from agentpool.utils.time_utils import ms_to_datetime
from agentpool_server.opencode_server.models import (
    AssistantMessage,
    FilePart as OpenCodeFilePart,
    ReasoningPart as OpenCodeReasoningPart,
    Session,
    TextPart as OpenCodeTextPart,
    TokenCache,
    Tokens,
    ToolPart as OpenCodeToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStatePending,
    ToolStateRunning,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from pydantic_ai import MultiModalContent
    from pydantic_ai.messages import UserContent

    from agentpool_server.opencode_server.models import (
        MessageInfo as OpenCodeMessage,
        Part as OpenCodePart,
    )


logger = get_logger(__name__)


def to_user_content(url: str, mime: str) -> MultiModalContent:
    if mime.startswith("image/"):
        mime_typ = mime if mime != "image/*" else None
        return ImageUrl(url=url, media_type=mime_typ)
    if mime.startswith("audio/"):
        mime_typ = mime if mime != "audio/*" else None
        return AudioUrl(url=url, media_type=mime_typ)
    if mime.startswith("video/"):
        mime_typ = mime if mime != "video/*" else None
        return VideoUrl(url=url, media_type=mime_typ)
    if mime == "application/pdf" or mime.startswith("application/"):
        return DocumentUrl(url=url, media_type=mime)
    # Unknown MIME type - treat as document
    return DocumentUrl(url=url, media_type=mime)


def convert_user_content_to_parts(
    content: str | Sequence[UserContent],
    message_id: str,
    session_id: str,
    part_counter_start: int,
) -> list[OpenCodeTextPart | OpenCodeFilePart]:
    """Convert UserContent to OpenCode parts.

    Args:
        content: User content (str or Sequence[UserContent])
        message_id: Message ID
        session_id: Session ID
        part_counter_start: Starting counter for part IDs

    Returns:
        List of OpenCode parts (TextPart for text/strings, FilePart for media)

    Note:
        CachePoint markers are lost in conversion (no OpenCode equivalent)
    """
    parts: list[OpenCodeTextPart | OpenCodeFilePart] = []
    part_counter = part_counter_start
    content_items = [content] if isinstance(content, str) else content
    # Sequence of UserContent
    for content_item in content_items:
        part_id = f"{message_id}-{part_counter}"
        part_counter += 1
        match content_item:
            case str():
                text_part = OpenCodeTextPart(
                    id=part_id,
                    session_id=session_id,
                    message_id=message_id,
                    text=content_item,
                )
                parts.append(text_part)
            case (
                ImageUrl(media_type=media_type, url=url)
                | AudioUrl(media_type=media_type, url=url)
                | DocumentUrl(media_type=media_type, url=url)
                | VideoUrl(media_type=media_type, url=url)
            ):
                file_part = OpenCodeFilePart(
                    id=part_id,
                    session_id=session_id,
                    message_id=message_id,
                    mime=media_type,
                    url=url,
                )
                parts.append(file_part)
            case BinaryContent(media_type=media_type, data=data):
                # Binary content - convert to data URL
                b64_data = base64.b64encode(data).decode("ascii")
                file_part = OpenCodeFilePart(
                    id=part_id,
                    session_id=session_id,
                    message_id=message_id,
                    mime=media_type,
                    url=f"data:{media_type};base64,{b64_data}",
                )
                parts.append(file_part)
        # CachePoint is just a marker for prompt caching - no data to store
    return parts


def extract_text_content(parts: list[OpenCodePart]) -> str:
    """Extract text content from parts for display.

    Groups consecutive reasoning parts into a single <thinking> block
    and only wraps them if there are also non-reasoning parts present.

    Args:
        parts: List of OpenCode parts

    Returns:
        Combined text content from all text and reasoning parts
    """
    text_segments: list[str] = []
    reasoning_segments: list[str] = []
    has_text = False

    for part in parts:
        match part:
            case OpenCodeTextPart(text=text) if text:
                has_text = True
                # Flush any accumulated reasoning before this text
                if reasoning_segments:
                    combined = "\n".join(reasoning_segments)
                    text_segments.append(f"<thinking>\n{combined}\n</thinking>")
                    reasoning_segments.clear()
                text_segments.append(text)
            case OpenCodeReasoningPart(text=text) if text:
                reasoning_segments.append(text)

    # Flush remaining reasoning
    if reasoning_segments:
        combined = "\n".join(reasoning_segments)
        if has_text:
            text_segments.append(f"<thinking>\n{combined}\n</thinking>")
        else:
            # Entire message is thinking — no need for wrapper tags
            text_segments.append(combined)

    return "\n".join(text_segments)


def user_message_to_pydantic(
    msg: UserMessage, parts: list[OpenCodePart], ts: datetime
) -> ModelRequest | None:
    user_content: list[UserContent] = []
    for part in parts:
        match part:
            case OpenCodeTextPart() if part.text:
                user_content.append(part.text)
            case OpenCodeFilePart(url=url, mime=mime):
                # Detect data URLs for BinaryContent
                if url.startswith("data:"):
                    # Parse data URL: data:mime;base64,data
                    if ";base64," in url:
                        mime_part, b64_data = url.split(";base64,", 1)
                        media_type = mime_part.replace("data:", "")
                        data = base64.b64decode(b64_data)
                        user_content.append(BinaryContent(data=data, media_type=media_type))
                    continue
                # Convert to appropriate URL type based on MIME
                content_item = to_user_content(url, mime)
                user_content.append(content_item)
    if user_content:
        user_part = UserPromptPart(content=user_content, timestamp=ts)
        return ModelRequest(parts=[user_part], timestamp=ts)
    return None


def build_pydantic_messages(
    msg: OpenCodeMessage,
    parts: list[OpenCodePart],
    timestamp: datetime,
) -> list[ModelRequest | ModelResponse]:
    """Build pydantic-ai ModelRequest and/or ModelResponse from OpenCode data.

    In OpenCode's model, assistant messages contain both tool calls AND their
    results in the same message. We split these into:
    - ModelResponse with ToolCallPart (the call)
    - ModelRequest with ToolReturnPart (the result)

    Args:
        msg: OpenCode message metadata
        parts: OpenCode message parts
        timestamp: Message timestamp

    Returns:
        List of pydantic-ai messages (ModelRequest and/or ModelResponse)
    """
    if isinstance(msg, UserMessage):
        if py_msg := user_message_to_pydantic(msg, parts, timestamp):
            return [py_msg]
        return []
    result: list[ModelRequest | ModelResponse] = []
    # Assistant message - may contain both tool calls and results
    response_parts: list[TextPart | ToolCallPart | ThinkingPart] = []
    tool_return_parts: list[ToolReturnPart] = []
    usage = RequestUsage(
        input_tokens=msg.tokens.input,
        output_tokens=msg.tokens.output,
        cache_read_tokens=msg.tokens.cache.read,
        cache_write_tokens=msg.tokens.cache.write,
    )
    for part in parts:
        match part:
            case OpenCodeTextPart(text=text) if text:
                response_parts.append(TextPart(content=text))
            case OpenCodeReasoningPart(text=text) if text:
                response_parts.append(ThinkingPart(content=text))
            case OpenCodeToolPart(
                call_id=call_id,
                tool=tool,
                state=ToolStateCompleted(output=output, input=input) as state,
            ) if output:
                # Add tool call to response
                tc_part = ToolCallPart(tool_name=tool, args=input or {}, tool_call_id=call_id)
                response_parts.append(tc_part)
                return_part = ToolReturnPart(
                    tool_name=tool,
                    content=state.output,
                    tool_call_id=call_id,
                    timestamp=timestamp,
                )
                tool_return_parts.append(return_part)
            case OpenCodeToolPart(
                call_id=call_id,
                tool=tool,
                state=ToolStatePending(input=input)
                | ToolStateRunning(input=input)
                | ToolStateError(input=input),
            ):
                tc_part = ToolCallPart(tool_name=tool, args=input or {}, tool_call_id=call_id)
                response_parts.append(tc_part)
    # Add the response if we have parts
    if response_parts:
        # AssistantMessage only has model_id, not model
        model_response = ModelResponse(
            parts=response_parts,
            usage=usage,
            model_name=msg.model_id,
            timestamp=timestamp,
        )
        result.append(model_response)
    # Add tool returns as a separate request (simulating user sending results back)
    if tool_return_parts:
        result.append(ModelRequest(parts=tool_return_parts, timestamp=timestamp))
    return result


def read_session(session_path: Path) -> Session | None:
    """Read session metadata from session JSON file. Returns None if read/parse fails."""
    if not session_path.exists():
        return None
    try:
        content = session_path.read_text(encoding="utf-8")
        return anyenv.load_json(content, return_type=Session)
    except anyenv.JsonLoadError as e:
        logger.warning("Failed to parse session", path=str(session_path), error=str(e))
        return None


def to_chat_message(
    msg: OpenCodeMessage,
    parts: list[OpenCodePart],
    session_id: str,
) -> ChatMessage[str]:
    """Convert OpenCode message + parts to ChatMessage."""
    timestamp = ms_to_datetime(msg.time.created)
    content = extract_text_content(parts)
    pydantic_messages = build_pydantic_messages(msg, parts, timestamp)
    cost_info = None
    provider_details = {}
    match msg:
        case AssistantMessage(
            tokens=Tokens(input=input, output=output, cache=TokenCache(read=read)),
        ):
            if (input_tokens := input + read) or output:
                usage = RunUsage(input_tokens=input_tokens, output_tokens=output)
                cost = Decimal(str(msg.cost)) if msg.cost else Decimal(0)
                cost_info = TokenCost(token_usage=usage, total_cost=cost)
            if msg.finish:
                provider_details["finish_reason"] = msg.finish

            return ChatMessage[str](
                content=content,
                session_id=session_id,
                role=msg.role,
                message_id=msg.id,
                name=msg.agent,
                model_name=msg.model_id,
                cost_info=cost_info,
                timestamp=timestamp,
                parent_id=msg.parent_id,
                messages=pydantic_messages,
                provider_details=provider_details,
            )

        case UserMessage(role=role, id=msg_id, agent=agent, model=model):
            return ChatMessage[str](
                content=content,
                session_id=session_id,
                role=role,
                message_id=msg_id,
                name=agent,
                model_name=model.model_id if model else None,
                cost_info=cost_info,
                timestamp=timestamp,
                parent_id=None,
                messages=pydantic_messages,
                provider_details=provider_details,
            )
