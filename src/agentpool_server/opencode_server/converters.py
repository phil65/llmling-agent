"""Converters between pydantic-ai/AgentPool and OpenCode message formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import uuid

from pydantic_ai import TextPart as PydanticTextPart, ToolCallPart as PydanticToolCallPart

from agentpool_server.opencode_server.models import (
    TextPart,
    TimeStartEnd,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStatePending,
    ToolStateRunning,
)
from agentpool_server.opencode_server.time_utils import now_ms


if TYPE_CHECKING:
    from pydantic_ai import ModelResponse, ToolReturnPart as PydanticToolReturnPart

    from agentpool.agents.events import (
        ToolCallCompleteEvent,
        ToolCallProgressEvent,
        ToolCallStartEvent,
    )
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
            time=TimeStartEnd(start=now_ms() - 1000, end=now_ms()),
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
            time=TimeStartEnd(start=now_ms()),
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
            time=TimeStartEnd(start=now_ms() - 1000, end=now_ms()),
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
) -> str | list[Any]:
    """Extract user prompt from OpenCode message parts.

    Converts OpenCode parts to pydantic-ai UserContent format:
    - Text parts become strings
    - File parts become ImageUrl, DocumentUrl, AudioUrl, VideoUrl, or BinaryContent

    Args:
        parts: List of OpenCode message parts

    Returns:
        Either a simple string (text-only) or a list of UserContent items
    """
    result: list[Any] = []

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
        return "\n".join(result)

    return result
