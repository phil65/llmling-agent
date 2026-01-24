"""ClaudeCodeAgent Exceptions."""

from __future__ import annotations


class ThinkingModeAlreadyConfiguredError(ValueError):
    """Raised when attempting to change thinking mode when max_thinking_tokens is configured."""

    def __init__(self) -> None:
        msg = (
            "Cannot change thinking mode: max_thinking_tokens is configured. "
            "The envvar MAX_THINKING_TOKENS takes precedence over the 'ultrathink' keyword."
        )
        super().__init__(msg)
