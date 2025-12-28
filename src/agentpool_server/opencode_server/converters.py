"""Converters between pydantic-ai/AgentPool and OpenCode message formats."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
import uuid

from agentpool_server.opencode_server.models import (
    TextPart,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStatePending,
    ToolStateRunning,
)


if TYPE_CHECKING:
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart as PydanticTextPart,
        ToolCallPart as PydanticToolCallPart,
        ToolReturnPart as PydanticToolReturnPart,
    )

    from agentpool.agents.events.events import (
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
        state=ToolStatePending(
            status="pending",
            title=f"Calling {part.tool_name}",
            input=part.args if isinstance(part.args, dict) else {},
        ),
    )


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

    if is_error:
        state = ToolStateError(
            status="error",
            title=f"Error in {part.tool_name}",
            error=str(content.get("error", "Unknown error")),
            input=existing_tool_part.state.input if existing_tool_part else {},
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
            input=existing_tool_part.state.input if existing_tool_part else {},
            output=output,
            time={"start": time.time() - 1, "end": time.time()},  # Approximate
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
    from pydantic_ai.messages import (
        TextPart as PydanticTextPart,
        ToolCallPart as PydanticToolCallPart,
    )

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
        state=ToolStatePending(
            status="pending",
            title=event.title or f"Calling {event.tool_name}",
            input=event.tool_input or {},
        ),
    )


def convert_tool_progress_event(
    event: ToolCallProgressEvent,
    existing_part: ToolPart,
) -> ToolPart:
    """Update ToolPart with progress from AgentPool ToolCallProgressEvent."""
    # Build output from progress content
    output_parts: list[str] = []
    for item in event.content:
        if hasattr(item, "text"):
            output_parts.append(item.text)
        elif hasattr(item, "data"):
            output_parts.append(item.data)

    output = "\n".join(output_parts) if output_parts else ""

    return ToolPart(
        id=existing_part.id,
        session_id=existing_part.session_id,
        message_id=existing_part.message_id,
        tool=existing_part.tool,
        call_id=existing_part.call_id,
        state=ToolStateRunning(
            status="running",
            title=event.title or existing_part.state.title,
            input=existing_part.state.input,
            output=output,
        ),
    )


def convert_tool_complete_event(
    event: ToolCallCompleteEvent,
    existing_part: ToolPart,
) -> ToolPart:
    """Update ToolPart with completion from AgentPool ToolCallCompleteEvent."""
    # Format the result
    result = event.result
    if isinstance(result, str):
        output = result
    elif isinstance(result, dict):
        import json

        output = json.dumps(result, indent=2)
    else:
        output = str(result) if result is not None else ""

    if event.error:
        state = ToolStateError(
            status="error",
            title=f"Error in {existing_part.tool}",
            error=str(event.error),
            input=existing_part.state.input,
        )
    else:
        state = ToolStateCompleted(
            status="completed",
            title=f"Completed {existing_part.tool}",
            input=existing_part.state.input,
            output=output,
            time={"start": time.time() - 1, "end": time.time()},
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


def convert_opencode_text_input(text: str) -> str:
    """Convert OpenCode text input to pydantic-ai format.

    For now this is a simple pass-through, but could handle
    special formatting or attachments in the future.
    """
    return text


def extract_user_prompt_from_parts(parts: list[dict[str, Any]]) -> str:
    """Extract user prompt text from OpenCode message parts."""
    text_parts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    return "\n".join(text_parts)
