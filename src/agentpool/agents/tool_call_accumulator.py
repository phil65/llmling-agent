"""Tool call accumulator for streaming tool arguments.

This module provides utilities for accumulating streamed tool call arguments
from LLM APIs that stream JSON arguments incrementally (like Anthropic's
input_json_delta or OpenAI's function call streaming).
"""

from __future__ import annotations

from typing import Any

import anyenv


def repair_partial_json(buffer: str) -> str:
    """Attempt to repair truncated JSON for preview purposes.

    Handles common truncation cases:
    - Unclosed strings
    - Missing closing braces/brackets
    - Trailing commas

    Args:
        buffer: Potentially incomplete JSON string

    Returns:
        Repaired JSON string (may still be invalid in edge cases)
    """
    if not buffer:
        return "{}"
    result = buffer.rstrip()
    # Check if we're in the middle of a string by counting unescaped quotes
    in_string = False
    i = 0
    while i < len(result):
        char = result[i]
        if char == "\\" and i + 1 < len(result):
            i += 2  # Skip escaped character
            continue
        if char == '"':
            in_string = not in_string
        i += 1

    # Close unclosed string
    if in_string:
        result += '"'

    # Remove trailing comma (invalid JSON)
    result = result.rstrip()
    if result.endswith(","):
        result = result[:-1]
    # Balance braces and brackets
    open_braces = result.count("{") - result.count("}")
    open_brackets = result.count("[") - result.count("]")
    result += "]" * max(0, open_brackets)
    result += "}" * max(0, open_braces)
    return result


class ToolCallAccumulator:
    """Accumulates streamed tool call arguments.

    LLM APIs stream tool call arguments as deltas. This class accumulates them
    and provides the complete arguments when the tool call ends, as well as
    best-effort partial argument parsing during streaming.

    Example:
        ```python
        accumulator = ToolCallAccumulator()

        # On content_block_start with tool_use
        accumulator.start("toolu_123", "write_file")

        # On input_json_delta events
        accumulator.add_args("toolu_123", '{"path": "/tmp/')
        accumulator.add_args("toolu_123", 'test.txt", "content"')
        accumulator.add_args("toolu_123", ': "hello"}')

        # Get partial args for UI preview (repairs incomplete JSON)
        partial = accumulator.get_partial_args("toolu_123")

        # On content_block_stop, get final parsed args
        name, args = accumulator.complete("toolu_123")
        ```
    """

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, Any]] = {}

    def start(self, tool_call_id: str, tool_name: str) -> None:
        """Start tracking a new tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
        """
        self._calls[tool_call_id] = {"name": tool_name, "args_buffer": ""}

    def add_args(self, tool_call_id: str, delta: str) -> None:
        """Add argument delta to a tool call.

        Args:
            tool_call_id: Tool call identifier
            delta: JSON string fragment to append
        """
        if tool_call_id in self._calls:
            self._calls[tool_call_id]["args_buffer"] += delta

    def complete(self, tool_call_id: str) -> tuple[str, dict[str, Any]] | None:
        """Complete a tool call and return (tool_name, parsed_args).

        Removes the tool call from tracking and returns the final parsed arguments.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Tuple of (tool_name, args_dict) or None if call not found
        """
        if tool_call_id not in self._calls:
            return None

        call_data = self._calls.pop(tool_call_id)
        args_str = call_data["args_buffer"]
        try:
            args = anyenv.load_json(args_str) if args_str else {}
        except anyenv.JsonLoadError:
            args = {"_raw": args_str}
        return call_data["name"], args

    def get_pending(self, tool_call_id: str) -> tuple[str, str] | None:
        """Get pending call data (tool_name, args_buffer) without completing.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Tuple of (tool_name, args_buffer) or None if not found
        """
        if tool_call_id not in self._calls:
            return None
        data = self._calls[tool_call_id]
        return data["name"], data["args_buffer"]

    def get_partial_args(self, tool_call_id: str) -> dict[str, Any]:
        """Get best-effort parsed args from incomplete JSON stream.

        Uses heuristics to complete truncated JSON for preview purposes.
        Handles unclosed strings, missing braces/brackets, and trailing commas.

        Args:
            tool_call_id: Tool call ID

        Returns:
            Partially parsed arguments or empty dict
        """
        if tool_call_id not in self._calls:
            return {}

        buffer = self._calls[tool_call_id]["args_buffer"]
        if not buffer:
            return {}

        # Try direct parse first
        try:
            result: dict[str, Any] = anyenv.load_json(buffer)
        except anyenv.JsonLoadError:
            pass
        else:
            return result

        # Try to repair truncated JSON
        try:
            repaired = repair_partial_json(buffer)
            result = anyenv.load_json(repaired)
        except anyenv.JsonLoadError:
            return {}
        else:
            return result

    def is_pending(self, tool_call_id: str) -> bool:
        """Check if a tool call is being tracked.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            True if the tool call is being accumulated
        """
        return tool_call_id in self._calls

    def get_tool_name(self, tool_call_id: str) -> str | None:
        """Get the tool name for a pending call.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Tool name or None if not found
        """
        if tool_call_id not in self._calls:
            return None
        name: str = self._calls[tool_call_id]["name"]
        return name

    def clear(self) -> None:
        """Clear all pending tool calls."""
        self._calls.clear()
