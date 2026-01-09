"""Simple converter for ChatMessage to OpenCode format for sharing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import uuid


if TYPE_CHECKING:
    from agentpool.messaging.messages import ChatMessage


def chat_message_to_opencode_simple(
    msg: ChatMessage,
    session_id: str,
) -> tuple[dict, list[dict]]:
    """Convert ChatMessage to OpenCode API format for sharing.

    Simpler than the full converter - just extracts text content without
    needing pydantic-ai message structure.

    Args:
        msg: ChatMessage to convert
        session_id: OpenCode session ID

    Returns:
        Tuple of (message_dict, parts_list) ready for OpenCode share API
    """
    message_id = msg.message_id
    created_ms = int(msg.timestamp.timestamp() * 1000)

    # Build message info
    message_info = {
        "id": message_id,
        "sessionID": session_id,
        "role": msg.role,
        "time": {"created": created_ms},
    }

    # Add assistant-specific fields
    if msg.role == "assistant":
        completed_ms = created_ms
        if msg.response_time:
            completed_ms = created_ms + int(msg.response_time * 1000)

        # Extract token/cost info - use msg.usage for tokens (matches server converter)
        # msg.usage has the raw RequestUsage from pydantic-ai
        usage = msg.usage
        if usage:
            input_tokens = usage.input_tokens or 0
            output_tokens = usage.output_tokens or 0
            cache_read = usage.cache_read_tokens or 0
            cache_write = usage.cache_write_tokens or 0
        else:
            input_tokens = output_tokens = cache_read = cache_write = 0

        tokens = {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": 0,
            "cache": {"read": cache_read, "write": cache_write},
        }

        # Get cost from cost_info if available
        cost = float(msg.cost_info.total_cost) if msg.cost_info else 0.0

        message_info.update({
            "time": {"created": created_ms, "completed": completed_ms},
            "parentID": msg.parent_id or "",
            "modelID": msg.model_name or "unknown",
            "providerID": msg.provider_name or "agentpool",
            "mode": "default",
            "agent": msg.name or "default",
            "path": {"cwd": "/tmp", "root": "/tmp"},
            "cost": cost,
            "tokens": tokens,
        })
    else:
        # User message
        message_info["agent"] = msg.name or "default"
        message_info["model"] = {
            "providerID": "agentpool",
            "modelID": "unknown",
        }

    # Extract text content
    if hasattr(msg, "text_content") and msg.text_content:
        text = msg.text_content
    elif isinstance(msg.content, str):
        text = msg.content
    else:
        text = str(msg.content)

    # Build parts - note ALL keys must be in the exact format OpenCode expects
    parts = []

    # For assistant messages, add step start
    if msg.role == "assistant":
        parts.append({
            "id": str(uuid.uuid4()),
            "type": "step-start",
            "messageID": message_id,
            "sessionID": session_id,
        })

    # Add text part
    parts.append({
        "id": str(uuid.uuid4()),
        "type": "text",
        "messageID": message_id,
        "sessionID": session_id,
        "text": text,
    })

    # For assistant messages, add step finish
    if msg.role == "assistant":
        parts.append({
            "id": str(uuid.uuid4()),
            "type": "step-finish",
            "messageID": message_id,
            "sessionID": session_id,
            "reason": "stop",
            "cost": message_info.get("cost", 0.0),
            "tokens": message_info.get("tokens", {
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            }),
        })

    return message_info, parts
