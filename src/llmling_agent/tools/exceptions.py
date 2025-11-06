"""Tool-related exceptions."""

from __future__ import annotations

from llmling_agent.utils.baseregistry import LLMLingError


class ToolError(LLMLingError):
    """Tool-related errors."""
