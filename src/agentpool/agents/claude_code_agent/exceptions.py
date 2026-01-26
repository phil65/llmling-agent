"""ClaudeCodeAgent Exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from clawd_code_sdk.types import AssistantMessage


class ThinkingModeAlreadyConfiguredError(ValueError):
    """Raised when attempting to change thinking mode when max_thinking_tokens is configured."""

    def __init__(self) -> None:
        msg = (
            "Cannot change thinking mode: max_thinking_tokens is configured. "
            "The envvar MAX_THINKING_TOKENS takes precedence over the 'ultrathink' keyword."
        )
        super().__init__(msg)


def raise_if_usage_limit_reached(message: AssistantMessage) -> None:
    """Raise RuntimeError if usage limit is reached."""
    from clawd_code_sdk.types import TextBlock

    if len(message.content) == 1 and isinstance(message.content[0], TextBlock):
        text_content = message.content[0].text
        if "You've hit your limit" in text_content and "resets" in text_content:
            raise RuntimeError(f"Claude Code usage limit reached: {text_content}")
