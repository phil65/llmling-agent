"""Shared agent exceptions."""

from __future__ import annotations

from collections.abc import Sequence


class AgentNotInitializedError(RuntimeError):
    """Raised when an agent is not initialized."""

    def __init__(self):
        super().__init__("Agent not initialized - use async context manager")


class UnknownCategoryError(ValueError):
    """Raised when an unknown category is encountered."""

    def __init__(self, category_id: str):
        msg = f"Unknown category: {category_id}. Available: permissions, model, thought_level"
        super().__init__(msg)


class UnknownModeError(ValueError):
    """Raised when an unknown mode is encountered."""

    def __init__(self, mode_id: str, available_modes: Sequence[str]):
        msg = f"Unknown mode: {mode_id}. Available: {', '.join(available_modes)}"
        super().__init__(msg)
