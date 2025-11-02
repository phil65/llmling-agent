"""Serialization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, TypeAdapter

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import ModelRequestPart, ModelResponsePart


logger = get_logger(__name__)

# Type adapter for serializing ModelResponsePart sequences
parts_adapter: TypeAdapter = TypeAdapter(
    list,
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def deserialize_parts(parts_json: str | None) -> Sequence[ModelResponsePart]:
    """Deserialize pydantic-ai message parts from JSON string.

    Args:
        parts_json: JSON string representation of parts or None if empty

    Returns:
        Sequence of ModelResponsePart objects, empty if deserialization fails
    """
    if not parts_json:
        return []

    try:
        # Deserialize using pydantic's JSON deserialization
        return parts_adapter.validate_json(parts_json.encode())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to deserialize message parts", error=e)
        return []  # Return empty list on failure


def serialize_parts(parts: Sequence[ModelResponsePart | ModelRequestPart]) -> str | None:
    """Serialize pydantic-ai message parts from ChatMessage.

    Args:
        parts: Sequence of ModelResponsePart from ChatMessage.parts

    Returns:
        JSON string representation of parts or None if empty
    """
    if not parts:
        return None

    try:
        # Convert parts to serializable format
        serializable_parts = []
        for part in parts:
            # Handle RetryPromptPart context serialization issues
            from pydantic_ai import RetryPromptPart

            if isinstance(part, RetryPromptPart) and isinstance(part.content, list):
                for content in part.content:
                    if isinstance(content, dict) and "ctx" in content:
                        content["ctx"] = {k: str(v) for k, v in content["ctx"].items()}
            serializable_parts.append(part)

        # Serialize using pydantic's JSON serialization
        return parts_adapter.dump_json(serializable_parts).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to serialize message parts", error=e)
        return str(parts)  # Fallback to string representation
