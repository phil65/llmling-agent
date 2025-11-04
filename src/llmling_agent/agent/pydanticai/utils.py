"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic_ai import ToolCallPart, ToolReturnPart

from llmling_agent.tools import ToolCallInfo


if TYPE_CHECKING:
    from pydantic_ai import ModelMessage

    from llmling_agent.tools.base import Tool


def get_tool_calls(
    messages: list[ModelMessage],
    tools: dict[str, Tool] | None = None,
    agent_name: str | None = None,
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        tools: Original Tool set to enrich ToolCallInfos with additional info
        agent_name: Name of the caller
    """
    tools = tools or {}
    parts = [part for message in messages for part in message.parts]
    call_parts = {
        part.tool_call_id: part
        for part in parts
        if isinstance(part, ToolCallPart) and part.tool_call_id
    }
    return [
        parts_to_tool_call_info(
            call_parts[part.tool_call_id],
            part,
            t.agent_name if (t := tools.get(part.tool_name)) else None,
            agent_name=agent_name,
        )
        for part in parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id in call_parts
    ]


def parts_to_tool_call_info(
    call_part: ToolCallPart,
    return_part: ToolReturnPart,
    agent_tool_name: str | None,
    agent_name: str | None = None,
) -> ToolCallInfo:
    """Convert matching tool call and return parts into a ToolCallInfo."""
    return ToolCallInfo(
        tool_name=call_part.tool_name,
        args=call_part.args_as_dict(),
        agent_name=agent_name or "UNSET",
        result=return_part.content,
        tool_call_id=call_part.tool_call_id or str(uuid4()),
        timestamp=return_part.timestamp,
        agent_tool_name=agent_tool_name,
    )
