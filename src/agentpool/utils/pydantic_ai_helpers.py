"""Helper utilities for working with pydantic-ai message types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import BaseToolCallPart


if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPartDelta


def safe_args_as_dict(
    part: BaseToolCallPart | ToolCallPartDelta,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safely extract args as dict from a tool call part.

    Models can return malformed JSON for tool arguments, especially during
    streaming when args are still being assembled. This helper catches parse
    errors and returns a fallback value.

    Args:
        part: A tool call part (complete or delta) with args to extract
        default: Value to return on parse failure. If None, returns {"_raw_args": ...}
                 with the original unparsed args.

    Returns:
        The parsed arguments dict, or a fallback on parse failure.
    """
    if not isinstance(part, BaseToolCallPart):
        # ToolCallPartDelta doesn't have args_as_dict
        if default is not None:
            return default
        raw = getattr(part, "args", None)
        return {"_raw_args": raw} if raw else {}
    try:
        return part.args_as_dict()
    except ValueError:
        # Model returned malformed JSON for tool args
        if default is not None:
            return default
        # Preserve raw args for debugging/inspection
        return {"_raw_args": part.args} if part.args else {}
