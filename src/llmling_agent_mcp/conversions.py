"""Conversions between internal and MCP types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import types


if TYPE_CHECKING:
    from llmling import BasePrompt, PromptParameter

    from llmling_agent.messaging import ChatMessage


def to_mcp_message(msg: ChatMessage) -> types.PromptMessage:
    """Convert internal PromptMessage to MCP PromptMessage."""
    role: types.Role = "assistant" if msg.role == "assistant" else "user"
    content = types.TextContent(type="text", text=msg.content)
    return types.PromptMessage(role=role, content=content)


def to_mcp_argument(arg: PromptParameter) -> types.PromptArgument:
    """Convert to MCP PromptArgument."""
    return types.PromptArgument(
        name=arg.name, description=arg.description, required=arg.required
    )


def to_mcp_prompt(prompt: BasePrompt) -> types.Prompt:
    """Convert to MCP Prompt."""
    if prompt.name is None:
        msg = "Prompt name not set. This should be set during registration."
        raise ValueError(msg)
    args = [to_mcp_argument(arg) for arg in prompt.arguments]
    return types.Prompt(name=prompt.name, description=prompt.description, arguments=args)
