"""ClaudeCodeAgent Exceptions."""

from __future__ import annotations


class ThinkingModeAlreadyConfiguredError(ValueError):
    """Raised when attempting to change thinking mode when max_thinking_tokens is configured."""

    def __init__(self):
        msg = (
            "Cannot change thinking mode: max_thinking_tokens is configured. "
            "The envvar MAX_THINKING_TOKENS takes precedence over the 'ultrathink' keyword."
        )
        super().__init__(msg)


class UnknownCategoryError(ValueError):
    """Raised when an unknown category is encountered."""

    def __init__(self, category_id: str):
        msg = f"Unknown category: {category_id}. Available: permissions, model, thought_level"
        super().__init__(msg)


class AgentNotInitializedError(RuntimeError):
    """Raised when an agent is not initialized."""

    def __init__(self):
        super().__init__("Agent not initialized - use async context manager")
