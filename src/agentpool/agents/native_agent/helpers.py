"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    BaseToolCallPart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartStartEvent,
    TextPart,
    ToolReturnPart,
)

from agentpool.agents.events import ToolCallCompleteEvent
from agentpool.agents.modes import ModeCategory, ModeInfo
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict


if TYPE_CHECKING:
    from tokonomics.model_discovery import ModelInfo

    from agentpool.agents.events import RichAgentStreamEvent


def process_tool_event(
    agent_name: str,
    event: RichAgentStreamEvent[Any],
    pending_tool_calls: dict[str, BaseToolCallPart],
    message_id: str,
) -> ToolCallCompleteEvent | None:
    """Process tool-related events and return combined event when complete.

    Args:
        agent_name: Name of the agent
        event: The streaming event to process
        pending_tool_calls: Dict tracking in-progress tool calls by ID
        message_id: Message ID for the combined event

    Returns:
        ToolCallCompleteEvent if a tool call completed, None otherwise
    """
    match event:
        case PartStartEvent(part=BaseToolCallPart() as tool_part):
            pending_tool_calls[tool_part.tool_call_id] = tool_part
        case FunctionToolCallEvent(part=tool_part):
            pending_tool_calls[tool_part.tool_call_id] = tool_part
        case FunctionToolResultEvent(tool_call_id=call_id) as result_event:
            if call_info := pending_tool_calls.pop(call_id, None):
                return ToolCallCompleteEvent(
                    tool_name=call_info.tool_name,
                    tool_call_id=call_id,
                    tool_input=safe_args_as_dict(call_info),
                    tool_result=result_event.result.content
                    if isinstance(result_event.result, ToolReturnPart)
                    else result_event.result,
                    agent_name=agent_name,
                    message_id=message_id,
                )
    return None


def extract_text_from_messages(messages: list[Any], include_interruption_note: bool = False) -> str:
    """Extract text content from pydantic-ai messages.

    Args:
        messages: List of ModelRequest/ModelResponse messages
        include_interruption_note: Whether to append interruption notice

    Returns:
        Concatenated text content from all ModelResponse TextParts
    """
    content = "".join(
        part.content
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, TextPart)
    )
    if include_interruption_note:
        if content:
            content += "\n\n"
        content += "[Request interrupted by user]"
    return content


def get_permission_category(current_mode: str) -> ModeCategory:
    return ModeCategory(
        id="permissions",
        name="Permissions",
        available_modes=[
            ModeInfo(
                id="default",
                name="Default",
                description="Require confirmation for tools marked as needing it",
                category_id="permissions",
            ),
            ModeInfo(
                id="acceptEdits",
                name="Accept Edits",
                description="Auto-approve all tool calls without confirmation",
                category_id="permissions",
            ),
        ],
        current_mode_id=current_mode,
        category="mode",
    )


def get_model_category(current_model: str, models: list[ModelInfo]) -> ModeCategory:
    return ModeCategory(
        id="model",
        name="Model",
        available_modes=[
            ModeInfo(
                id=m.id,
                name=m.name or m.id,
                description=m.description or "",
                category_id="model",
            )
            for m in models
        ],
        current_mode_id=current_model,
        category="model",
    )
