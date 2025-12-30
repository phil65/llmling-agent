"""Token breakdown utilities for analyzing context window usage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import (
        ModelMessage,
    )
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


@dataclass
class TokenUsage:
    """Single item's token count."""

    token_count: int
    label: str


@dataclass
class RunTokenUsage:
    """Token usage for a single agent run."""

    run_id: str | None
    token_count: int
    request_count: int


@dataclass
class TokenBreakdown:
    """Complete token breakdown of context."""

    total_tokens: int

    system_prompts: list[TokenUsage]
    tool_definitions: list[TokenUsage]
    runs: list[RunTokenUsage]

    approximate: bool

    @property
    def system_prompts_tokens(self) -> int:
        return sum(t.token_count for t in self.system_prompts)

    @property
    def tool_definitions_tokens(self) -> int:
        return sum(t.token_count for t in self.tool_definitions)

    @property
    def conversation_tokens(self) -> int:
        return sum(r.token_count for r in self.runs)


def _normalize_tool_schema(tool: ToolDefinition | dict[str, Any]) -> dict[str, Any]:
    """Convert a ToolDefinition or dict to a consistent dict format."""
    if isinstance(tool, ToolDefinition):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_json_schema,
        }
    return tool


def _count_tokens_tiktoken(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens using tiktoken as a fallback."""
    try:
        import tiktoken
    except ImportError:
        # Rough approximation: ~4 chars per token
        return len(text) // 4

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def _extract_system_prompts(messages: Sequence[ModelMessage]) -> list[str]:
    """Extract all system prompt contents from messages."""
    prompts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    prompts.append(part.content)  # noqa: PERF401
    return prompts


def _group_messages_by_run(
    messages: Sequence[ModelMessage],
) -> dict[str | None, list[ModelMessage]]:
    """Group messages by their run_id."""
    groups: dict[str | None, list[ModelMessage]] = defaultdict(list)
    for message in messages:
        run_id = message.run_id
        groups[run_id].append(message)
    return dict(groups)


def _messages_to_text(messages: Sequence[ModelMessage]) -> str:
    """Convert messages to a text representation for token counting."""
    text_parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for request_part in message.parts:
                if hasattr(request_part, "content") and isinstance(request_part.content, str):
                    text_parts.append(request_part.content)  # noqa: PERF401
        elif isinstance(message, ModelResponse):
            if text := message.text:
                text_parts.append(text)
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    # Tool call arguments
                    args = part.args
                    if isinstance(args, str):
                        text_parts.append(args)
                    elif args:
                        text_parts.append(json.dumps(args))
    return "\n".join(text_parts)


async def get_token_breakdown(
    model: Model,
    messages: Sequence[ModelMessage],
    tool_schemas: Sequence[ToolDefinition | dict[str, Any]] | None = None,
    model_settings: ModelSettings | None = None,
) -> TokenBreakdown:
    """Get a breakdown of token usage by component.

    Uses model.count_tokens() if available, falls back to tiktoken.

    Args:
        model: The model to use for token counting.
        messages: The message history to analyze.
        tool_schemas: Tool definitions or raw JSON schemas.
        model_settings: Optional model settings.

    Returns:
        A TokenBreakdown with detailed token usage by component.
    """
    tool_schemas = tool_schemas or []
    approximate = False
    model_name = model.model_name or "gpt-4"

    # Try to use model.count_tokens(), fall back to tiktoken
    async def count_tokens_for_messages(msgs: Sequence[ModelMessage]) -> int:
        nonlocal approximate
        try:
            # Build minimal ModelRequestParameters for counting
            params = ModelRequestParameters()
            usage = await model.count_tokens(list(msgs), model_settings, params)
        except NotImplementedError:
            approximate = True
            return _count_tokens_tiktoken(_messages_to_text(msgs), model_name)
        else:
            return usage.input_tokens

    # Extract and count system prompts
    system_prompt_contents = _extract_system_prompts(messages)
    system_prompt_usages: list[TokenUsage] = []
    for i, content in enumerate(system_prompt_contents):
        token_count = _count_tokens_tiktoken(content, model_name)
        label = content[:50] + "..." if len(content) > 50 else content  # noqa: PLR2004
        system_prompt_usages.append(
            TokenUsage(token_count=token_count, label=f"System prompt {i + 1}: {label}")
        )
    # Mark as approximate since we're using tiktoken for individual prompts
    if system_prompt_usages:
        approximate = True

    # Count tool definition tokens
    tool_usages: list[TokenUsage] = []
    for tool in tool_schemas:
        schema = _normalize_tool_schema(tool)
        schema_text = json.dumps(schema)
        token_count = _count_tokens_tiktoken(schema_text, model_name)
        tool_name = schema.get("name", "unknown")
        tool_usages.append(TokenUsage(token_count=token_count, label=tool_name))
    if tool_usages:
        approximate = True

    # Group messages by run and count tokens per run
    run_groups = _group_messages_by_run(messages)
    run_usages: list[RunTokenUsage] = []
    for run_id, run_messages in run_groups.items():
        token_count = await count_tokens_for_messages(run_messages)
        request_count = sum(1 for m in run_messages if isinstance(m, ModelRequest))
        run_usages.append(
            RunTokenUsage(
                run_id=run_id,
                token_count=token_count,
                request_count=request_count,
            )
        )

    # Calculate total
    total = (
        sum(u.token_count for u in system_prompt_usages)
        + sum(u.token_count for u in tool_usages)
        + sum(r.token_count for r in run_usages)
    )

    return TokenBreakdown(
        total_tokens=total,
        system_prompts=system_prompt_usages,
        tool_definitions=tool_usages,
        runs=run_usages,
        approximate=approximate,
    )
