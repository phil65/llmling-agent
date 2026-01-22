"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    BaseToolCallPart,
    BaseToolReturnPart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartStartEvent,
    TextPart,
)

from agentpool.agents.events import ToolCallCompleteEvent
from agentpool.agents.modes import ModeCategory, ModeInfo
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict


if TYPE_CHECKING:
    from tokonomics.model_discovery import ModelInfo

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool_config.nodes import ToolConfirmationMode


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
    # Note: BuiltinToolCallEvent/BuiltinToolResultEvent are deprecated.
    # Both function and builtin tools use PartStartEvent with BaseToolCallPart/BaseToolReturnPart.
    match event:
        case (
            PartStartEvent(part=BaseToolCallPart() as tool_part)
            | FunctionToolCallEvent(part=tool_part)
        ):
            pending_tool_calls[tool_part.tool_call_id] = tool_part
        case (
            PartStartEvent(part=BaseToolReturnPart(tool_call_id=call_id, content=content))
            | FunctionToolResultEvent(
                result=BaseToolReturnPart(tool_call_id=call_id, content=content)
            )
        ):
            if call_info := pending_tool_calls.pop(call_id, None):
                return ToolCallCompleteEvent(
                    tool_name=call_info.tool_name,
                    tool_call_id=call_id,
                    tool_input=safe_args_as_dict(call_info),
                    tool_result=content,
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


def get_permission_category(current_mode: ToolConfirmationMode) -> ModeCategory:
    """Get permission mode category using native ToolConfirmationMode values."""
    return ModeCategory(
        id="mode",
        name="Tool Confirmation",
        available_modes=[
            ModeInfo(
                id="always",
                name="Always",
                description="Always require confirmation for all tools",
                category_id="mode",
            ),
            ModeInfo(
                id="never",
                name="Never",
                description="Never require confirmation (auto-approve all)",
                category_id="mode",
            ),
            ModeInfo(
                id="per_tool",
                name="Per Tool",
                description="Require confirmation only for tools marked as needing it",
                category_id="mode",
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
