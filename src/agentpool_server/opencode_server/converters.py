"""Converters between pydantic-ai/AgentPool and OpenCode message formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import uuid

from pydantic_ai import (
    TextPart as PydanticTextPart,
    ToolCallPart as PydanticToolCallPart,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolReturnPart as PydanticToolReturnPart,
    UserPromptPart,
)

from agentpool_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageTime,
    MessageWithParts,
    TextPart,
    TimeStart,
    TimeStartEnd,
    TimeStartEndCompacted,
    Tokens,
    TokensCache,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStatePending,
    ToolStateRunning,
    UserMessage,
)
from agentpool_server.opencode_server.models.common import TimeCreated
from agentpool_server.opencode_server.models.message import UserMessageModel
from agentpool_server.opencode_server.models.parts import (
    StepFinishPart,
    StepFinishTokens,
    StepStartPart,
    TimeStartEndOptional,
    TokenCache,
)
from agentpool_server.opencode_server.time_utils import now_ms


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import (
        UserContent,
    )

    from agentpool.agents.events import (
        ToolCallCompleteEvent,
        ToolCallProgressEvent,
        ToolCallStartEvent,
    )
    from agentpool.messaging.messages import ChatMessage
    from agentpool_server.opencode_server.models import Part


def generate_part_id() -> str:
    """Generate a unique part ID."""
    return str(uuid.uuid4())


# =============================================================================
# Pydantic-AI to OpenCode Converters
# =============================================================================


def convert_pydantic_text_part(
    part: PydanticTextPart,
    session_id: str,
    message_id: str,
) -> TextPart:
    """Convert a pydantic-ai TextPart to OpenCode TextPart."""
    return TextPart(
        id=part.id or generate_part_id(),
        session_id=session_id,
        message_id=message_id,
        text=part.content,
    )


def convert_pydantic_tool_call_part(
    part: PydanticToolCallPart,
    session_id: str,
    message_id: str,
) -> ToolPart:
    """Convert a pydantic-ai ToolCallPart to OpenCode ToolPart (pending state)."""
    # Tool call started - create pending state
    return ToolPart(
        id=generate_part_id(),
        session_id=session_id,
        message_id=message_id,
        tool=part.tool_name,
        call_id=part.tool_call_id,
        state=ToolStatePending(status="pending"),
    )


def _get_input_from_state(
    state: ToolStatePending | ToolStateRunning | ToolStateCompleted | ToolStateError,
) -> dict[str, Any]:
    """Extract input from any tool state type."""
    if hasattr(state, "input") and state.input is not None:
        return state.input
    return {}


def convert_pydantic_tool_return_part(
    part: PydanticToolReturnPart,
    session_id: str,
    message_id: str,
    existing_tool_part: ToolPart | None = None,
) -> ToolPart:
    """Convert a pydantic-ai ToolReturnPart to OpenCode ToolPart (completed state)."""
    # Determine if it's an error or success based on content
    content = part.content
    is_error = isinstance(content, dict) and content.get("error")

    existing_input = _get_input_from_state(existing_tool_part.state) if existing_tool_part else {}

    if is_error:
        state: ToolStateCompleted | ToolStateError = ToolStateError(
            status="error",
            error=str(content.get("error", "Unknown error")),
            input=existing_input,
            time=TimeStartEnd(start=now_ms() - 1000, end=now_ms()),
        )
    else:
        # Format output for display
        if isinstance(content, str):
            output = content
        elif isinstance(content, dict):
            import json

            output = json.dumps(content, indent=2)
        else:
            output = str(content)

        state = ToolStateCompleted(
            status="completed",
            title=f"Completed {part.tool_name}",
            input=existing_input,
            output=output,
            time=TimeStartEndCompacted(start=now_ms() - 1000, end=now_ms()),
        )

    return ToolPart(
        id=existing_tool_part.id if existing_tool_part else generate_part_id(),
        session_id=session_id,
        message_id=message_id,
        tool=part.tool_name,
        call_id=part.tool_call_id,
        state=state,
    )


def convert_model_response_to_parts(
    response: ModelResponse,
    session_id: str,
    message_id: str,
) -> list[Part]:
    """Convert a pydantic-ai ModelResponse to OpenCode Parts."""
    parts: list[Part] = []

    for part in response.parts:
        if isinstance(part, PydanticTextPart):
            parts.append(convert_pydantic_text_part(part, session_id, message_id))
        elif isinstance(part, PydanticToolCallPart):
            parts.append(convert_pydantic_tool_call_part(part, session_id, message_id))
        # Other part types (ThinkingPart, FilePart) can be added as needed

    return parts


# =============================================================================
# AgentPool Event to OpenCode State Converters
# =============================================================================


def convert_tool_start_event(
    event: ToolCallStartEvent,
    session_id: str,
    message_id: str,
) -> ToolPart:
    """Convert AgentPool ToolCallStartEvent to OpenCode ToolPart."""
    return ToolPart(
        id=generate_part_id(),
        session_id=session_id,
        message_id=message_id,
        tool=event.tool_name,
        call_id=event.tool_call_id,
        state=ToolStatePending(status="pending"),
    )


def _get_title_from_state(
    state: ToolStatePending | ToolStateRunning | ToolStateCompleted | ToolStateError,
) -> str:
    """Extract title from any tool state type."""
    return getattr(state, "title", "")


def convert_tool_progress_event(
    event: ToolCallProgressEvent,
    existing_part: ToolPart,
) -> ToolPart:
    """Update ToolPart with progress from AgentPool ToolCallProgressEvent."""
    # ToolStateRunning doesn't have output field, progress is indicated by title
    return ToolPart(
        id=existing_part.id,
        session_id=existing_part.session_id,
        message_id=existing_part.message_id,
        tool=existing_part.tool,
        call_id=existing_part.call_id,
        state=ToolStateRunning(
            status="running",
            time=TimeStart(start=now_ms()),
            title=event.title or _get_title_from_state(existing_part.state),
            input=_get_input_from_state(existing_part.state),
        ),
    )


def convert_tool_complete_event(
    event: ToolCallCompleteEvent,
    existing_part: ToolPart,
) -> ToolPart:
    """Update ToolPart with completion from AgentPool ToolCallCompleteEvent."""
    # Format the result
    result = event.tool_result
    if isinstance(result, str):
        output = result
    elif isinstance(result, dict):
        import json

        output = json.dumps(result, indent=2)
    else:
        output = str(result) if result is not None else ""

    existing_input = _get_input_from_state(existing_part.state)

    # ToolCallCompleteEvent doesn't have error field - check result for error indication
    if isinstance(result, dict) and result.get("error"):
        state: ToolStateCompleted | ToolStateError = ToolStateError(
            status="error",
            error=str(result.get("error", "Unknown error")),
            input=existing_input,
            time=TimeStartEnd(start=now_ms() - 1000, end=now_ms()),
        )
    else:
        state = ToolStateCompleted(
            status="completed",
            title=f"Completed {existing_part.tool}",
            input=existing_input,
            output=output,
            time=TimeStartEndCompacted(start=now_ms() - 1000, end=now_ms()),
        )

    return ToolPart(
        id=existing_part.id,
        session_id=existing_part.session_id,
        message_id=existing_part.message_id,
        tool=existing_part.tool,
        call_id=existing_part.call_id,
        state=state,
    )


# =============================================================================
# OpenCode to Pydantic-AI Converters (for input)
# =============================================================================


def _convert_file_part_to_user_content(part: dict[str, Any]) -> Any:
    """Convert an OpenCode FilePartInput to pydantic-ai MultiModalContent.

    Supports:
    - Images (image/*) -> ImageUrl or BinaryContent
    - Documents (application/pdf, text/*) -> DocumentUrl or BinaryContent
    - Audio (audio/*) -> AudioUrl or BinaryContent
    - Video (video/*) -> VideoUrl or BinaryContent

    Args:
        part: OpenCode file part with mime, url, and optional filename

    Returns:
        Appropriate pydantic-ai content type
    """
    from pydantic_ai.messages import (
        AudioUrl,
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        VideoUrl,
    )

    mime = part.get("mime", "")
    url = part.get("url", "")

    # Handle data: URIs - convert to BinaryContent
    if url.startswith("data:"):
        return BinaryContent.from_data_uri(url)

    # Handle regular URLs or file paths based on mime type
    if mime.startswith("image/"):
        return ImageUrl(url=url)
    if mime.startswith("audio/"):
        return AudioUrl(url=url)
    if mime.startswith("video/"):
        return VideoUrl(url=url)
    if mime.startswith(("application/pdf", "text/")):
        return DocumentUrl(url=url)

    # Fallback: treat as document
    return DocumentUrl(url=url)


def extract_user_prompt_from_parts(
    parts: list[dict[str, Any]],
) -> str | Sequence[UserContent]:
    """Extract user prompt from OpenCode message parts.

    Converts OpenCode parts to pydantic-ai UserContent format:
    - Text parts become strings
    - File parts become ImageUrl, DocumentUrl, AudioUrl, VideoUrl, or BinaryContent

    Args:
        parts: List of OpenCode message parts

    Returns:
        Either a simple string (text-only) or a list of UserContent items
    """
    result: list[UserContent] = []

    for part in parts:
        part_type = part.get("type")

        if part_type == "text":
            text = part.get("text", "")
            if text:
                result.append(text)

        elif part_type == "file":
            content = _convert_file_part_to_user_content(part)
            result.append(content)

    # If only text parts, join them as a single string for simplicity
    if all(isinstance(item, str) for item in result):
        return "\n".join(result)  # type: ignore[arg-type]

    return result


# =============================================================================
# ChatMessage <-> OpenCode MessageWithParts Converters
# =============================================================================


def _datetime_to_ms(dt: Any) -> int:
    """Convert datetime to milliseconds timestamp."""
    from datetime import datetime

    if isinstance(dt, datetime):
        return int(dt.timestamp() * 1000)
    return now_ms()


def _ms_to_datetime(ms: int) -> Any:
    """Convert milliseconds timestamp to datetime."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def chat_message_to_opencode(  # noqa: PLR0915
    msg: ChatMessage[Any],
    session_id: str,
    working_dir: str = "",
    agent_name: str = "default",
    model_id: str = "unknown",
    provider_id: str = "agentpool",
) -> MessageWithParts:
    """Convert a ChatMessage to OpenCode MessageWithParts.

    Args:
        msg: The ChatMessage to convert
        session_id: OpenCode session ID
        working_dir: Working directory for path context
        agent_name: Name of the agent
        model_id: Model identifier
        provider_id: Provider identifier

    Returns:
        OpenCode MessageWithParts with appropriate info and parts
    """
    message_id = msg.message_id
    created_ms = _datetime_to_ms(msg.timestamp)

    parts: list[Part] = []

    # Track tool calls by ID for pairing with returns
    tool_calls: dict[str, ToolPart] = {}

    if msg.role == "user":
        # User message
        info: UserMessage | AssistantMessage = UserMessage(
            id=message_id,
            session_id=session_id,
            time=TimeCreated(created=created_ms),
            agent=agent_name,
            model=UserMessageModel(provider_id=provider_id, model_id=model_id),
        )

        # Extract text from user prompt parts
        for model_msg in msg.messages:
            if isinstance(model_msg, ModelRequest):
                for part in model_msg.parts:
                    if isinstance(part, UserPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            text = content
                        else:
                            # Multi-modal content - extract text parts
                            text = " ".join(str(c) for c in content if isinstance(c, str))
                        if text:
                            parts.append(
                                TextPart(
                                    id=generate_part_id(),
                                    message_id=message_id,
                                    session_id=session_id,
                                    text=text,
                                    time=TimeStartEndOptional(start=created_ms),
                                )
                            )
    else:
        # Assistant message
        completed_ms = created_ms
        if msg.response_time:
            completed_ms = created_ms + int(msg.response_time * 1000)

        # Extract token usage
        tokens = Tokens(
            input=msg.usage.input_tokens or 0 if msg.usage else 0,
            output=msg.usage.output_tokens or 0 if msg.usage else 0,
            reasoning=0,
            cache=TokensCache(
                read=msg.usage.cache_read_tokens or 0 if msg.usage else 0,
                write=msg.usage.cache_write_tokens or 0 if msg.usage else 0,
            ),
        )

        info = AssistantMessage(
            id=message_id,
            session_id=session_id,
            parent_id="",  # Would need to track parent user message
            model_id=msg.model_name or model_id,
            provider_id=msg.provider_name or provider_id,
            mode="default",
            agent=agent_name,
            path=MessagePath(cwd=working_dir, root=working_dir),
            time=MessageTime(created=created_ms, completed=completed_ms),
            tokens=tokens,
            cost=int(msg.cost_info.total_cost * 1_000_000) if msg.cost_info else 0,
            finish=msg.finish_reason,
        )

        # Add step start
        parts.append(
            StepStartPart(
                id=generate_part_id(),
                message_id=message_id,
                session_id=session_id,
            )
        )

        # Process all model messages to extract parts
        for model_msg in msg.messages:
            if isinstance(model_msg, ModelResponse):
                for p in model_msg.parts:
                    if isinstance(p, PydanticTextPart):
                        parts.append(
                            TextPart(
                                id=p.id or generate_part_id(),
                                message_id=message_id,
                                session_id=session_id,
                                text=p.content,
                                time=TimeStartEndOptional(start=created_ms, end=completed_ms),
                            )
                        )
                    elif isinstance(p, PydanticToolCallPart):
                        # Create tool part in pending/running state
                        from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict

                        tool_input = safe_args_as_dict(p)
                        tool_part = ToolPart(
                            id=generate_part_id(),
                            message_id=message_id,
                            session_id=session_id,
                            tool=p.tool_name,
                            call_id=p.tool_call_id or generate_part_id(),
                            state=ToolStateRunning(
                                status="running",
                                time=TimeStart(start=created_ms),
                                input=tool_input,
                                title=f"Running {p.tool_name}",
                            ),
                        )
                        tool_calls[p.tool_call_id or ""] = tool_part
                        parts.append(tool_part)

            elif isinstance(model_msg, ModelRequest):
                # Check for tool returns in requests (they come after responses)
                for part in model_msg.parts:
                    if isinstance(part, PydanticToolReturnPart):
                        call_id = part.tool_call_id or ""
                        existing = tool_calls.get(call_id)

                        # Format output
                        content = part.content
                        if isinstance(content, str):
                            output = content
                        elif isinstance(content, dict):
                            import json

                            output = json.dumps(content, indent=2)
                        else:
                            output = str(content) if content is not None else ""

                        # Check for error
                        is_error = isinstance(content, dict) and "error" in content

                        if existing:
                            # Update existing tool part with completion
                            existing_input = _get_input_from_state(existing.state)
                            if is_error:
                                existing.state = ToolStateError(
                                    status="error",
                                    error=str(content.get("error", "Unknown error")),
                                    input=existing_input,
                                    time=TimeStartEnd(start=created_ms, end=completed_ms),
                                )
                            else:
                                existing.state = ToolStateCompleted(
                                    status="completed",
                                    title=f"Completed {part.tool_name}",
                                    input=existing_input,
                                    output=output,
                                    time=TimeStartEndCompacted(start=created_ms, end=completed_ms),
                                )
                        else:
                            # Orphan return - create completed tool part
                            state: ToolStateCompleted | ToolStateError
                            if is_error:
                                state = ToolStateError(
                                    status="error",
                                    error=str(content.get("error", "Unknown error")),
                                    input={},
                                    time=TimeStartEnd(start=created_ms, end=completed_ms),
                                )
                            else:
                                state = ToolStateCompleted(
                                    status="completed",
                                    title=f"Completed {part.tool_name}",
                                    input={},
                                    output=output,
                                    time=TimeStartEndCompacted(start=created_ms, end=completed_ms),
                                )
                            parts.append(
                                ToolPart(
                                    id=generate_part_id(),
                                    message_id=message_id,
                                    session_id=session_id,
                                    tool=part.tool_name,
                                    call_id=call_id,
                                    state=state,
                                )
                            )

        # Add step finish
        parts.append(
            StepFinishPart(
                id=generate_part_id(),
                message_id=message_id,
                session_id=session_id,
                reason=msg.finish_reason or "stop",
                cost=int(msg.cost_info.total_cost * 1_000_000) if msg.cost_info else 0,
                tokens=StepFinishTokens(
                    input=tokens.input,
                    output=tokens.output,
                    reasoning=tokens.reasoning,
                    cache=TokenCache(read=tokens.cache.read, write=tokens.cache.write),
                ),
            )
        )

    return MessageWithParts(info=info, parts=parts)


def opencode_to_chat_message(
    msg: MessageWithParts,
    conversation_id: str | None = None,
) -> ChatMessage[str]:
    """Convert OpenCode MessageWithParts to ChatMessage.

    Args:
        msg: OpenCode message with parts
        conversation_id: Optional conversation ID override

    Returns:
        ChatMessage with pydantic-ai model messages
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse
    from pydantic_ai.usage import RequestUsage

    from agentpool.messaging.messages import ChatMessage

    info = msg.info
    message_id = info.id
    session_id = info.session_id

    # Determine role and extract timing
    if isinstance(info, UserMessage):
        role = "user"
        created_ms = info.time.created
        model_name = info.model.model_id if info.model else None
        provider_name = info.model.provider_id if info.model else None
        usage = RequestUsage()
        finish_reason = None
    else:
        role = "assistant"
        created_ms = info.time.created
        model_name = info.model_id
        provider_name = info.provider_id
        usage = RequestUsage(
            input_tokens=info.tokens.input,
            output_tokens=info.tokens.output,
            cache_read_tokens=info.tokens.cache.read,
            cache_write_tokens=info.tokens.cache.write,
        )
        finish_reason = info.finish

    timestamp = _ms_to_datetime(created_ms)

    # Build model messages from parts
    model_messages: list[ModelRequest | ModelResponse] = []

    if role == "user":
        # Collect text parts into a user prompt
        text_content = [part.text for part in msg.parts if isinstance(part, TextPart)]
        content = "\n".join(text_content) if text_content else ""
        model_messages.append(
            ModelRequest(
                parts=[UserPromptPart(content=content)],
                instructions=None,
            )
        )
    else:
        # Assistant message - collect response parts and tool interactions
        response_parts: list[Any] = []
        tool_returns: list[PydanticToolReturnPart] = []

        for part in msg.parts:
            if isinstance(part, TextPart):
                response_parts.append(PydanticTextPart(content=part.text, id=part.id))
            elif isinstance(part, ToolPart):
                # Create tool call part

                tool_input = _get_input_from_state(part.state)
                response_parts.append(
                    PydanticToolCallPart(
                        tool_name=part.tool,
                        tool_call_id=part.call_id,
                        args=tool_input,
                    )
                )

                # If completed/error, also create tool return
                if isinstance(part.state, ToolStateCompleted):
                    tool_returns.append(
                        PydanticToolReturnPart(
                            tool_name=part.tool,
                            tool_call_id=part.call_id,
                            content=part.state.output,
                        )
                    )
                elif isinstance(part.state, ToolStateError):
                    tool_returns.append(
                        PydanticToolReturnPart(
                            tool_name=part.tool,
                            tool_call_id=part.call_id,
                            content={"error": part.state.error},
                        )
                    )
            # Skip StepStartPart, StepFinishPart, FilePart for now

        if response_parts:
            model_messages.append(
                ModelResponse(
                    parts=response_parts,
                    usage=usage,
                    model_name=model_name,
                    timestamp=timestamp,
                )
            )

        # Add tool returns as a follow-up request if any
        if tool_returns:
            model_messages.append(
                ModelRequest(
                    parts=tool_returns,
                    instructions=None,
                )
            )

    # Extract content for the ChatMessage
    content = ""
    for part in msg.parts:
        if isinstance(part, TextPart):
            content = part.text
            break

    return ChatMessage(
        content=content,
        role=role,  # type: ignore[arg-type]
        message_id=message_id,
        conversation_id=conversation_id or session_id,
        timestamp=timestamp,
        messages=model_messages,
        usage=usage,
        model_name=model_name,
        provider_name=provider_name,
        finish_reason=finish_reason,  # type: ignore[arg-type]
    )


def chat_messages_to_opencode(
    messages: list[ChatMessage[Any]],
    session_id: str,
    working_dir: str = "",
    agent_name: str = "default",
    model_id: str = "unknown",
    provider_id: str = "agentpool",
) -> list[MessageWithParts]:
    """Convert a list of ChatMessages to OpenCode format.

    Args:
        messages: List of ChatMessages to convert
        session_id: OpenCode session ID
        working_dir: Working directory for path context
        agent_name: Name of the agent
        model_id: Model identifier
        provider_id: Provider identifier

    Returns:
        List of OpenCode MessageWithParts
    """
    return [
        chat_message_to_opencode(
            msg,
            session_id=session_id,
            working_dir=working_dir,
            agent_name=agent_name,
            model_id=model_id,
            provider_id=provider_id,
        )
        for msg in messages
    ]


def opencode_to_chat_messages(
    messages: list[MessageWithParts],
    conversation_id: str | None = None,
) -> list[ChatMessage[str]]:
    """Convert a list of OpenCode messages to ChatMessages.

    Args:
        messages: List of OpenCode MessageWithParts
        conversation_id: Optional conversation ID override

    Returns:
        List of ChatMessages
    """
    return [opencode_to_chat_message(msg, conversation_id=conversation_id) for msg in messages]
