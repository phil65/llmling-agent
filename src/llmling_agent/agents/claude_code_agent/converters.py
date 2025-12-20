"""Claude Agent SDK to native event converters.

This module provides conversion from Claude Agent SDK message types to native
llmling-agent streaming events, enabling ClaudeCodeAgent to yield the same
event types as native agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import PartDeltaEvent
from pydantic_ai.messages import TextPartDelta, ThinkingPartDelta

from llmling_agent.agents.events import ToolCallCompleteEvent, ToolCallStartEvent


if TYPE_CHECKING:
    from claude_agent_sdk import ContentBlock, Message, ToolUseBlock

    from llmling_agent.agents.events import RichAgentStreamEvent


def content_block_to_event(
    block: ContentBlock,
    index: int = 0,
) -> RichAgentStreamEvent[Any] | None:
    """Convert a Claude SDK ContentBlock to a streaming event.

    Args:
        block: Claude SDK content block
        index: Part index for the event

    Returns:
        Corresponding streaming event, or None if not mappable
    """
    from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock

    match block:
        case TextBlock(text=text):
            return PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=text))
        case ThinkingBlock(thinking=thinking):
            return PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=thinking))
        case ToolUseBlock(id=tool_id, name=name, input=input_data):
            return ToolCallStartEvent(
                tool_call_id=tool_id,
                tool_name=name,
                title=name,
                kind="other",
                raw_input=input_data,
            )
        case _:
            return None


def claude_message_to_events(
    message: Message,
    agent_name: str = "",
    pending_tool_calls: dict[str, ToolUseBlock] | None = None,
) -> list[RichAgentStreamEvent[Any]]:
    """Convert a Claude SDK Message to a list of streaming events.

    Args:
        message: Claude SDK message (UserMessage, AssistantMessage, etc.)
        agent_name: Name of the agent for event attribution
        pending_tool_calls: Dict to track tool calls awaiting results

    Returns:
        List of corresponding streaming events
    """
    from claude_agent_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock

    events: list[RichAgentStreamEvent[Any]] = []

    match message:
        case AssistantMessage(content=content):
            for idx, block in enumerate(content):
                # Track tool use blocks for later pairing with results
                if isinstance(block, ToolUseBlock) and pending_tool_calls is not None:
                    pending_tool_calls[block.id] = block

                # Handle tool results - pair with pending tool call
                if isinstance(block, ToolResultBlock) and pending_tool_calls is not None:
                    tool_use = pending_tool_calls.pop(block.tool_use_id, None)
                    if tool_use:
                        events.append(
                            ToolCallCompleteEvent(
                                tool_name=tool_use.name,
                                tool_call_id=block.tool_use_id,
                                tool_input=tool_use.input,
                                tool_result=block.content,
                                agent_name=agent_name,
                                message_id="",
                            )
                        )
                    continue

                # Convert other blocks to events
                if event := content_block_to_event(block, index=idx):
                    events.append(event)

        case _:
            # UserMessage, SystemMessage, ResultMessage - no events to emit
            pass

    return events


def extract_text_from_message(message: Message) -> str:
    """Extract text content from a Claude SDK message.

    Args:
        message: Claude SDK message

    Returns:
        Concatenated text content from all text blocks
    """
    from claude_agent_sdk import AssistantMessage, TextBlock

    if not isinstance(message, AssistantMessage):
        return ""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))
