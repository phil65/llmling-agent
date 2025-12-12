"""Composable message compaction pipeline for managing conversation history.

This module provides a pipeline-based approach to compacting and transforming
pydantic-ai message history. Each step in the pipeline operates on the message
sequence and can filter, truncate, summarize, or transform messages.

Example:
    ```python
    from llmling_agent.messaging.compaction import (
        CompactionPipeline,
        FilterThinking,
        TruncateToolOutputs,
        KeepLastMessages,
    )

    # Programmatic usage
    pipeline = CompactionPipeline(steps=[
        FilterThinking(),
        TruncateToolOutputs(max_length=1000),
        KeepLastMessages(count=10),
    ])
    compacted = await pipeline.apply(messages)

    # Or via config (for YAML)
    config = CompactionPipelineConfig(steps=[
        FilterThinkingConfig(),
        TruncateToolOutputsConfig(max_length=1000),
        KeepLastMessagesConfig(count=10),
    ])
    pipeline = config.build()
    ```

YAML configuration example:
    ```yaml
    compaction:
      steps:
        - type: filter_thinking
        - type: truncate_tool_outputs
          max_length: 1000
        - type: keep_last
          count: 10
        - type: summarize
          model: openai:gpt-4o-mini
          threshold: 20
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast

from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent, ModelRequestPart
    from pydantic_ai.messages import ModelResponsePart
    from tokonomics import ModelName

# Type aliases
ModelMessage = ModelRequest | ModelResponse
MessageSequence = Sequence[ModelMessage]


class CompactionStep(ABC):
    """Base class for message compaction steps.

    Each step transforms a sequence of messages into a (potentially) smaller
    or modified sequence. Steps can be composed into a pipeline.
    """

    @abstractmethod
    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        """Apply this compaction step to the message sequence.

        Args:
            messages: The input message sequence to transform.

        Returns:
            The transformed message sequence.
        """
        ...

    def __or__(self, other: CompactionStep) -> CompactionPipeline:
        """Compose two steps into a pipeline using the | operator."""
        return CompactionPipeline(steps=[self, other])


@dataclass
class CompactionPipeline(CompactionStep):
    """A pipeline of compaction steps applied in sequence.

    Steps are applied left-to-right, with each step receiving the output
    of the previous step.
    """

    steps: list[CompactionStep] = field(default_factory=list)

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        """Apply all steps in sequence."""
        result: list[ModelMessage] = list(messages)
        for step in self.steps:
            result = await step.apply(result)
        return result

    def __or__(self, other: CompactionStep) -> CompactionPipeline:
        """Add another step to the pipeline."""
        if isinstance(other, CompactionPipeline):
            return CompactionPipeline(steps=[*self.steps, *other.steps])
        return CompactionPipeline(steps=[*self.steps, other])

    def __ior__(self, other: CompactionStep) -> Self:
        """Add a step in place."""
        if isinstance(other, CompactionPipeline):
            self.steps.extend(other.steps)
        else:
            self.steps.append(other)
        return self


# =============================================================================
# Filter Steps - Remove or filter parts of messages
# =============================================================================


@dataclass
class FilterThinking(CompactionStep):
    """Remove all thinking parts from model responses.

    Thinking parts can consume significant context space without providing
    value in subsequent interactions.
    """

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for msg in messages:
            match msg:
                case ModelResponse(parts=parts) if any(isinstance(p, ThinkingPart) for p in parts):
                    filtered_parts = [p for p in parts if not isinstance(p, ThinkingPart)]
                    if filtered_parts:  # Only include if there are remaining parts
                        result.append(replace(msg, parts=filtered_parts))
                case _:
                    result.append(msg)
        return result


@dataclass
class FilterRetryPrompts(CompactionStep):
    """Remove retry prompt parts from requests.

    Retry prompts are typically not needed after the conversation has moved on.
    """

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for msg in messages:
            match msg:
                case ModelRequest(parts=parts) if any(
                    isinstance(p, RetryPromptPart) for p in parts
                ):
                    filtered_parts = [p for p in parts if not isinstance(p, RetryPromptPart)]
                    if filtered_parts:
                        result.append(replace(msg, parts=filtered_parts))
                case _:
                    result.append(msg)
        return result


@dataclass
class FilterBinaryContent(CompactionStep):
    """Remove binary content (images, audio, etc.) from messages.

    Useful when you want to keep only text content for context efficiency.
    """

    keep_references: bool = False
    """If True, replace binary with a placeholder text describing what was there."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        from pydantic_ai.messages import BinaryContent

        result: list[ModelMessage] = []
        for msg in messages:
            match msg:
                case ModelRequest(parts=parts):
                    filtered_parts: list[ModelRequestPart | ModelResponsePart] = []
                    for part in parts:
                        if isinstance(part, UserPromptPart):
                            if isinstance(part.content, list):
                                new_content: list[Any] = []
                                for item in part.content:
                                    if isinstance(item, BinaryContent):
                                        if self.keep_references:
                                            new_content.append(f"[Binary: {item.media_type}]")
                                    else:
                                        new_content.append(item)
                                if new_content:
                                    filtered_parts.append(replace(part, content=new_content))
                            else:
                                filtered_parts.append(part)
                        else:
                            filtered_parts.append(part)
                    if filtered_parts:
                        result.append(replace(msg, parts=cast(Sequence[Any], filtered_parts)))
                case _:
                    result.append(msg)
        return result


@dataclass
class FilterToolCalls(CompactionStep):
    """Filter tool calls by name.

    Can be used to remove specific tool calls that are not relevant
    for future context (e.g., debugging tools, one-time lookups).
    """

    exclude_tools: list[str] = field(default_factory=list)
    """Tool names to exclude from the history."""

    include_only: list[str] | None = None
    """If set, only keep these tools (overrides exclude_tools)."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        from pydantic_ai.messages import ToolCallPart

        def should_keep(tool_name: str) -> bool:
            if self.include_only is not None:
                return tool_name in self.include_only
            return tool_name not in self.exclude_tools

        result: list[ModelMessage] = []
        excluded_call_ids: set[str] = set()

        for msg in messages:
            match msg:
                case ModelResponse(parts=parts):
                    filtered_parts: list[ModelRequestPart | ModelResponsePart] = []
                    for part in parts:
                        if isinstance(part, ToolCallPart):
                            if should_keep(part.tool_name):
                                filtered_parts.append(part)
                            else:
                                excluded_call_ids.add(part.tool_call_id)
                        else:
                            filtered_parts.append(part)
                    if filtered_parts:
                        result.append(replace(msg, parts=cast(Sequence[Any], filtered_parts)))

                case ModelRequest(parts=parts):
                    # Also filter corresponding tool returns
                    filtered_parts = [
                        p
                        for p in parts
                        if not (
                            isinstance(p, ToolReturnPart) and p.tool_call_id in excluded_call_ids
                        )
                    ]
                    if filtered_parts:
                        result.append(replace(msg, parts=cast(Sequence[Any], filtered_parts)))

                case _:
                    result.append(msg)

        return result


@dataclass
class FilterEmptyMessages(CompactionStep):
    """Remove messages that have no meaningful content.

    Cleans up the history by removing empty or near-empty messages.
    """

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for msg in messages:
            match msg:
                case ModelRequest(parts=parts) | ModelResponse(parts=parts):
                    if any(_part_has_content(p) for p in parts):
                        result.append(msg)
                case _:
                    result.append(msg)
        return result


def _part_has_content(part: Any) -> bool:
    """Check if a message part has meaningful content."""
    # Use has_content if available (TextPart, ThinkingPart, etc.)
    if hasattr(part, "has_content"):
        return part.has_content()  # type: ignore[no-any-return]
    # For UserPromptPart, check content directly
    if isinstance(part, UserPromptPart):
        content = part.content
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(content)
        return content is not None
    # For ToolReturnPart, check content
    if isinstance(part, ToolReturnPart):
        return part.content is not None
    # Default: assume has content
    return True


# =============================================================================
# Truncation Steps - Shorten content while preserving structure
# =============================================================================


@dataclass
class TruncateToolOutputs(CompactionStep):
    """Truncate large tool outputs to a maximum length.

    Tool outputs can sometimes be very large (e.g., file contents, API responses).
    This step truncates them while preserving the beginning of the content.
    """

    max_length: int = 2000
    """Maximum length for tool output content."""

    suffix: str = "\n... [truncated]"
    """Suffix to append when content is truncated."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for msg in messages:
            match msg:
                case ModelRequest(parts=parts):
                    new_parts: list[ModelRequestPart | ModelResponsePart] = []
                    for part in parts:
                        if isinstance(part, ToolReturnPart):
                            content = part.content
                            if isinstance(content, str) and len(content) > self.max_length:
                                truncated = content[: self.max_length - len(self.suffix)]
                                new_parts.append(replace(part, content=truncated + self.suffix))
                            else:
                                new_parts.append(part)
                        else:
                            new_parts.append(part)
                    result.append(replace(msg, parts=cast(Sequence[Any], new_parts)))
                case _:
                    result.append(msg)
        return result


@dataclass
class TruncateTextParts(CompactionStep):
    """Truncate long text parts in responses.

    Useful for limiting very long model responses in the context.
    """

    max_length: int = 5000
    """Maximum length for text content."""

    suffix: str = "\n... [truncated]"
    """Suffix to append when content is truncated."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        result: list[ModelMessage] = []
        for msg in messages:
            match msg:
                case ModelResponse(parts=parts):
                    new_parts: list[ModelRequestPart | ModelResponsePart] = []
                    for part in parts:
                        if isinstance(part, TextPart) and len(part.content) > self.max_length:
                            truncated = part.content[: self.max_length - len(self.suffix)]
                            new_parts.append(replace(part, content=truncated + self.suffix))
                        else:
                            new_parts.append(part)
                    result.append(replace(msg, parts=cast(Sequence[Any], new_parts)))
                case _:
                    result.append(msg)
        return result


# =============================================================================
# Selection Steps - Keep subsets of messages
# =============================================================================


@dataclass
class KeepLastMessages(CompactionStep):
    """Keep only the last N messages.

    A simple sliding window approach to context management.
    Messages are counted as request/response pairs when `count_pairs` is True.
    """

    count: int = 10
    """Number of messages (or pairs) to keep."""

    count_pairs: bool = True
    """If True, count request/response pairs instead of individual messages."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        if not messages:
            return []

        if not self.count_pairs:
            return list(messages[-self.count :])

        # Count pairs (each request+response = 1 pair)
        pairs: list[list[ModelMessage]] = []
        current_pair: list[ModelMessage] = []

        for msg in messages:
            current_pair.append(msg)
            if isinstance(msg, ModelResponse):
                pairs.append(current_pair)
                current_pair = []

        # Don't forget incomplete pair at the end
        if current_pair:
            pairs.append(current_pair)

        # Keep last N pairs
        kept_pairs = pairs[-self.count :]
        return [msg for pair in kept_pairs for msg in pair]


@dataclass
class KeepFirstMessages(CompactionStep):
    """Keep only the first N messages.

    Useful for keeping initial context/instructions while discarding
    middle conversation.
    """

    count: int = 2
    """Number of messages to keep from the beginning."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        return list(messages[: self.count])


@dataclass
class KeepFirstAndLast(CompactionStep):
    """Keep first N and last M messages, discarding the middle.

    Useful for preserving initial context while maintaining recent history.
    """

    first_count: int = 2
    """Number of messages to keep from the beginning."""

    last_count: int = 5
    """Number of messages to keep from the end."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        msg_list = list(messages)
        if len(msg_list) <= self.first_count + self.last_count:
            return msg_list

        first = msg_list[: self.first_count]
        last = msg_list[-self.last_count :]
        return first + last


# =============================================================================
# Advanced Steps - Token-aware and LLM-based
# =============================================================================


@dataclass
class TokenBudget(CompactionStep):
    """Keep messages that fit within a token budget.

    Works backwards from most recent, adding messages until the budget
    is exhausted. Requires tokonomics for token counting.
    """

    max_tokens: int = 4000
    """Maximum number of tokens to allow."""

    model: str = "gpt-4o"
    """Model to use for token counting."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        try:
            import tokonomics
        except ImportError:
            # Fall back to character-based estimation
            return await self._apply_char_estimate(messages)

        result: list[ModelMessage] = []
        total_tokens = 0

        # Process from most recent to oldest
        for msg in reversed(messages):
            # Estimate tokens for this message
            text = _extract_text_content(msg)
            token_count = tokonomics.count_tokens(text, self.model)

            if total_tokens + token_count > self.max_tokens:
                break

            result.insert(0, msg)
            total_tokens += token_count

        return result

    async def _apply_char_estimate(self, messages: MessageSequence) -> list[ModelMessage]:
        """Fallback using character-based token estimation (4 chars ≈ 1 token)."""
        result: list[ModelMessage] = []
        total_chars = 0
        max_chars = self.max_tokens * 4

        for msg in reversed(messages):
            text = _extract_text_content(msg)
            char_count = len(text)

            if total_chars + char_count > max_chars:
                break

            result.insert(0, msg)
            total_chars += char_count

        return result


@dataclass
class Summarize(CompactionStep):
    """Summarize older messages using an LLM.

    When the message count exceeds the threshold, older messages are
    summarized into a single message while recent ones are kept intact.
    """

    model: str = "openai:gpt-4o-mini"
    """Model to use for summarization."""

    threshold: int = 15
    """Minimum message count before summarization kicks in."""

    keep_recent: int = 5
    """Number of recent messages to keep unsummarized."""

    summary_prompt: str = (
        "Summarize the following conversation history concisely, "
        "preserving key information, decisions, and context that may be "
        "relevant for continuing the conversation:\n\n{conversation}"
    )
    """Prompt template for summarization. Use {conversation} placeholder."""

    _agent: Agent[None, str] | None = field(default=None, repr=False)

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        if len(messages) <= self.threshold:
            return list(messages)

        # Split into messages to summarize and messages to keep
        to_summarize = list(messages[: -self.keep_recent])
        to_keep = list(messages[-self.keep_recent :])

        # Format conversation for summarization
        conversation_text = _format_conversation(to_summarize)

        # Get or create summarization agent
        agent = await self._get_agent()

        # Generate summary
        prompt = self.summary_prompt.format(conversation=conversation_text)
        result = await agent.run(prompt)

        # Create summary message
        summary_request = ModelRequest(
            parts=[UserPromptPart(content=f"[Conversation Summary]\n{result.output}")]
        )

        return [summary_request, *to_keep]

    async def _get_agent(self) -> Agent[None, str]:
        """Get or create the summarization agent."""
        if self._agent is None:
            from pydantic_ai import Agent

            self._agent = Agent(model=self.model, output_type=str)
        return self._agent


# =============================================================================
# Conditional Steps - Apply steps based on conditions
# =============================================================================


@dataclass
class ConditionalStep(CompactionStep):
    """Apply a step only when a condition is met."""

    step: CompactionStep
    """The step to conditionally apply."""

    condition: Callable[[MessageSequence], bool]
    """Function that returns True if the step should be applied."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        if self.condition(messages):
            return await self.step.apply(messages)
        return list(messages)


@dataclass
class WhenMessageCountExceeds(CompactionStep):
    """Apply a step only when message count exceeds a threshold."""

    step: CompactionStep
    """The step to conditionally apply."""

    threshold: int = 20
    """Message count threshold."""

    async def apply(self, messages: MessageSequence) -> list[ModelMessage]:
        if len(messages) > self.threshold:
            return await self.step.apply(messages)
        return list(messages)


# =============================================================================
# Configuration Models - For YAML/JSON configuration
# =============================================================================


class FilterThinkingConfig(BaseModel):
    """Configuration for FilterThinking step."""

    type: Literal["filter_thinking"] = "filter_thinking"

    def build(self) -> FilterThinking:
        return FilterThinking()


class FilterRetryPromptsConfig(BaseModel):
    """Configuration for FilterRetryPrompts step."""

    type: Literal["filter_retry_prompts"] = "filter_retry_prompts"

    def build(self) -> FilterRetryPrompts:
        return FilterRetryPrompts()


class FilterBinaryContentConfig(BaseModel):
    """Configuration for FilterBinaryContent step."""

    type: Literal["filter_binary"] = "filter_binary"
    keep_references: bool = False

    def build(self) -> FilterBinaryContent:
        return FilterBinaryContent(keep_references=self.keep_references)


class FilterToolCallsConfig(BaseModel):
    """Configuration for FilterToolCalls step."""

    type: Literal["filter_tools"] = "filter_tools"
    exclude_tools: list[str] = Field(default_factory=list)
    include_only: list[str] | None = None

    def build(self) -> FilterToolCalls:
        return FilterToolCalls(exclude_tools=self.exclude_tools, include_only=self.include_only)


class FilterEmptyMessagesConfig(BaseModel):
    """Configuration for FilterEmptyMessages step."""

    type: Literal["filter_empty"] = "filter_empty"

    def build(self) -> FilterEmptyMessages:
        return FilterEmptyMessages()


class TruncateToolOutputsConfig(BaseModel):
    """Configuration for TruncateToolOutputs step."""

    type: Literal["truncate_tool_outputs"] = "truncate_tool_outputs"
    max_length: int = 2000
    suffix: str = "\n... [truncated]"

    def build(self) -> TruncateToolOutputs:
        return TruncateToolOutputs(max_length=self.max_length, suffix=self.suffix)


class TruncateTextPartsConfig(BaseModel):
    """Configuration for TruncateTextParts step."""

    type: Literal["truncate_text"] = "truncate_text"
    max_length: int = 5000
    suffix: str = "\n... [truncated]"

    def build(self) -> TruncateTextParts:
        return TruncateTextParts(max_length=self.max_length, suffix=self.suffix)


class KeepLastMessagesConfig(BaseModel):
    """Configuration for KeepLastMessages step."""

    type: Literal["keep_last"] = "keep_last"
    count: int = 10
    count_pairs: bool = True

    def build(self) -> KeepLastMessages:
        return KeepLastMessages(count=self.count, count_pairs=self.count_pairs)


class KeepFirstMessagesConfig(BaseModel):
    """Configuration for KeepFirstMessages step."""

    type: Literal["keep_first"] = "keep_first"
    count: int = 2

    def build(self) -> KeepFirstMessages:
        return KeepFirstMessages(count=self.count)


class KeepFirstAndLastConfig(BaseModel):
    """Configuration for KeepFirstAndLast step."""

    type: Literal["keep_first_last"] = "keep_first_last"
    first_count: int = 2
    last_count: int = 5

    def build(self) -> KeepFirstAndLast:
        return KeepFirstAndLast(first_count=self.first_count, last_count=self.last_count)


class TokenBudgetConfig(BaseModel):
    """Configuration for TokenBudget step."""

    type: Literal["token_budget"] = "token_budget"
    max_tokens: int = 4000
    model: str = "gpt-4o"

    def build(self) -> TokenBudget:
        return TokenBudget(max_tokens=self.max_tokens, model=self.model)


class SummarizeConfig(BaseModel):
    """Configuration for Summarize step."""

    type: Literal["summarize"] = "summarize"
    model: str = "openai:gpt-4o-mini"
    threshold: int = 15
    keep_recent: int = 5
    summary_prompt: str | None = None

    def build(self) -> Summarize:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "threshold": self.threshold,
            "keep_recent": self.keep_recent,
        }
        if self.summary_prompt:
            kwargs["summary_prompt"] = self.summary_prompt
        return Summarize(**kwargs)


class WhenMessageCountExceedsConfig(BaseModel):
    """Configuration for WhenMessageCountExceeds wrapper."""

    type: Literal["when_count_exceeds"] = "when_count_exceeds"
    threshold: int = 20
    step: "CompactionStepConfig"

    def build(self) -> WhenMessageCountExceeds:
        return WhenMessageCountExceeds(step=self.step.build(), threshold=self.threshold)


# Union of all config types with discriminator
CompactionStepConfig = Annotated[
    FilterThinkingConfig
    | FilterRetryPromptsConfig
    | FilterBinaryContentConfig
    | FilterToolCallsConfig
    | FilterEmptyMessagesConfig
    | TruncateToolOutputsConfig
    | TruncateTextPartsConfig
    | KeepLastMessagesConfig
    | KeepFirstMessagesConfig
    | KeepFirstAndLastConfig
    | TokenBudgetConfig
    | SummarizeConfig
    | WhenMessageCountExceedsConfig,
    Field(discriminator="type"),
]

# Update forward reference
WhenMessageCountExceedsConfig.model_rebuild()


class CompactionPipelineConfig(BaseModel):
    """Configuration for a complete compaction pipeline.

    Example YAML:
        ```yaml
        compaction:
          steps:
            - type: filter_thinking
            - type: truncate_tool_outputs
              max_length: 1000
            - type: keep_last
              count: 10
        ```
    """

    steps: list[CompactionStepConfig] = Field(default_factory=list)
    """Ordered list of compaction steps to apply."""

    def build(self) -> CompactionPipeline:
        """Build a CompactionPipeline from this configuration."""
        return CompactionPipeline(steps=[step.build() for step in self.steps])


# =============================================================================
# Preset Pipelines - Common configurations
# =============================================================================


def minimal_context() -> CompactionPipeline:
    """Create a pipeline that aggressively minimizes context.

    Removes thinking, truncates outputs, and keeps only recent messages.
    """
    return CompactionPipeline(
        steps=[
            FilterThinking(),
            FilterRetryPrompts(),
            TruncateToolOutputs(max_length=500),
            KeepLastMessages(count=5),
        ]
    )


def balanced_context() -> CompactionPipeline:
    """Create a balanced pipeline for general use.

    Removes thinking, moderately truncates, keeps reasonable history.
    """
    return CompactionPipeline(
        steps=[
            FilterThinking(),
            TruncateToolOutputs(max_length=2000),
            TruncateTextParts(max_length=5000),
            KeepLastMessages(count=15),
        ]
    )


def summarizing_context(model: ModelName | str = "openai:gpt-4o-mini") -> CompactionPipeline:
    """Create a pipeline that summarizes older messages.

    Best for long conversations where context needs to be preserved.
    """
    return CompactionPipeline(
        steps=[
            FilterThinking(),
            TruncateToolOutputs(max_length=1000),
            Summarize(model=model, threshold=20, keep_recent=8),
        ]
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_text_content(msg: ModelMessage) -> str:
    """Extract text content from a message for token counting."""
    parts_text: list[str] = []

    match msg:
        case ModelRequest(parts=parts) | ModelResponse(parts=parts):
            for part in parts:
                match part:
                    case TextPart(content=content):
                        parts_text.append(content)
                    case ThinkingPart(content=content):
                        parts_text.append(content)
                    case UserPromptPart(content=content):
                        if isinstance(content, str):
                            parts_text.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, str):
                                    parts_text.append(item)  # noqa: PERF401
                    case ToolReturnPart(content=content):
                        if isinstance(content, str):
                            parts_text.append(content)
                        elif content is not None:
                            parts_text.append(str(content))

    return "\n".join(parts_text)


def _format_conversation(messages: Sequence[ModelMessage]) -> str:
    """Format messages as a readable conversation for summarization."""
    lines: list[str] = []

    for msg in messages:
        match msg:
            case ModelRequest(parts=parts):
                for request_part in parts:
                    match request_part:
                        case UserPromptPart(content=content):
                            text = content if isinstance(content, str) else str(content)
                            lines.append(f"User: {text}")
                        case ToolReturnPart(tool_name=name, content=content):
                            lines.append(f"Tool Result ({name}): {content}")
            case ModelResponse(parts=parts):
                for response_part in parts:
                    match response_part:
                        case TextPart(content=content):
                            lines.append(f"Assistant: {content}")

    return "\n\n".join(lines)
