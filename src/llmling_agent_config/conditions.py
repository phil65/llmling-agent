"""Condition configuration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema

from llmling_agent.utils.inspection import execute
from llmling_agent.utils.now import get_now


if TYPE_CHECKING:
    from llmling_agent.talk.registry import EventContext


class ConnectionCondition(Schema):
    """Base class for connection control conditions."""

    type: str = Field(init=False)
    """Discriminator for condition types."""

    name: str | None = None
    """Optional name for the condition for referencing."""

    model_config = ConfigDict(frozen=True)

    async def check(self, context: EventContext) -> bool:
        """Check if condition is met."""
        raise NotImplementedError


class Jinja2Condition(ConnectionCondition):
    """Evaluate condition using Jinja2 template."""

    type: Literal["jinja2"] = Field("jinja2", init=False)
    """Jinja2 template-based condition."""

    template: str
    """Jinja2 template to evaluate."""

    async def check(self, ctx: EventContext) -> bool:
        from jinjarope import Environment

        env = Environment(trim_blocks=True, lstrip_blocks=True, enable_async=True)
        template = env.from_string(self.template)
        result = await template.render_async(ctx=ctx, now=get_now())
        return result.strip().lower() == "true" or bool(result)


class WordMatchCondition(ConnectionCondition):
    """Disconnect when word/phrase is found in message."""

    type: Literal["word_match"] = Field("word_match", init=False)
    """Word-comparison-based condition."""

    words: list[str]
    """Words or phrases to match in messages."""

    case_sensitive: bool = False
    """Whether to match case-sensitively."""

    mode: Literal["any", "all"] = "any"
    """Match mode:
    - any: Trigger if any word matches
    - all: Require all words to match
    """

    async def check(self, context: EventContext) -> bool:
        """Check if message contains specified words."""
        text = str(context.message.content)
        if not self.case_sensitive:
            text = text.lower()
            words = [w.lower() for w in self.words]
        else:
            words = self.words

        matches = [w in text for w in words]
        return all(matches) if self.mode == "all" else any(matches)


class MessageCountCondition(ConnectionCondition):
    """Disconnect after N messages."""

    type: Literal["message_count"] = Field("message_count", init=False)
    """Message-count-based condition."""

    max_messages: int = Field(gt=0)
    """Maximum number of messages before triggering."""

    count_mode: Literal["total", "per_agent"] = "total"
    """How to count messages:
    - total: All messages in conversation
    - per_agent: Messages from each agent separately
    """

    async def check(self, context: EventContext) -> bool:
        """Check if message count threshold is reached."""
        if self.count_mode == "total":
            return context.stats.message_count >= self.max_messages

        # Count per agent
        messages = [m for m in context.stats.messages if m.name == context.message.name]
        return len(messages) >= self.max_messages


class TimeCondition(ConnectionCondition):
    """Disconnect after time period."""

    type: Literal["time"] = Field("time", init=False)
    """Time-based condition."""

    duration: timedelta
    """How long the connection should stay active."""

    async def check(self, context: EventContext) -> bool:
        """Check if time duration has elapsed."""
        elapsed = get_now() - context.stats.start_time
        return elapsed >= self.duration


class TokenThresholdCondition(ConnectionCondition):
    """Disconnect after token threshold is reached."""

    type: Literal["token_threshold"] = Field("token_threshold", init=False)
    """Type discriminator."""

    max_tokens: int = Field(gt=0)
    """Maximum number of tokens allowed."""

    count_type: Literal["total", "prompt", "completion"] = "total"
    """What tokens to count:
    - total: All tokens used
    - prompt: Only prompt tokens
    - completion: Only completion tokens
    """

    async def check(self, context: EventContext) -> bool:
        """Check if token threshold is reached."""
        if not context.message.cost_info:
            return False

        match self.count_type:
            case "total":
                return context.stats.token_count >= self.max_tokens
            case "prompt":
                return context.message.cost_info.token_usage["prompt"] >= self.max_tokens
            case "completion":
                return (
                    context.message.cost_info.token_usage["completion"] >= self.max_tokens
                )
            case _:
                msg = f"Unknown count type: {self.count_type}"
                raise ValueError(msg)


class CostCondition(ConnectionCondition):
    """Stop when cost threshold is reached."""

    type: Literal["cost"] = Field("cost", init=False)
    """Cost-based condition."""

    max_cost: float = Field(gt=0.0)
    """Maximum cost in USD."""

    async def check(self, context: EventContext) -> bool:
        """Check if cost limit is reached."""
        return context.stats.total_cost >= self.max_cost


class CostLimitCondition(ConnectionCondition):
    """Disconnect when cost limit is reached."""

    type: Literal["cost_limit"] = Field("cost_limit", init=False)
    """Cost-limit condition."""

    max_cost: float = Field(gt=0.0)
    """Maximum cost in USD before triggering."""

    async def check(self, context: EventContext) -> bool:
        """Check if cost limit is reached."""
        if not context.message.cost_info:
            return False
        return float(context.message.cost_info.total_cost) >= self.max_cost


class CallableCondition(ConnectionCondition):
    """Custom predicate function."""

    type: Literal["callable"] = Field("callable", init=False)
    """Condition based on an import path pointing to a predicate."""

    predicate: ImportString[Callable[..., bool | Awaitable[bool]]]
    """Function to evaluate condition:
    Args:
        message: Current message being processed
        stats: Current connection statistics
    Returns:
        Whether condition is met
    """

    async def check(self, context: EventContext) -> bool:
        """Execute predicate function."""
        return await execute(self.predicate, context.message, context.stats)


class AndCondition(ConnectionCondition):
    """Require all conditions to be met."""

    type: Literal["and"] = Field("and", init=False)
    """Condition to AND-combine multiple conditions."""

    conditions: list[ConnectionCondition]
    """List of conditions to check."""

    async def check(self, context: EventContext) -> bool:
        """Check if all conditions are met."""
        results = [await c.check(context) for c in self.conditions]
        return all(results)


class OrCondition(ConnectionCondition):
    """Require any condition to be met."""

    type: Literal["or"] = Field("or", init=False)
    """Condition to OR-combine multiple conditions."""

    conditions: list[ConnectionCondition]
    """List of conditions to check."""

    async def check(self, context: EventContext) -> bool:
        """Check if any condition is met."""
        results = [await c.check(context) for c in self.conditions]
        return any(results)


# Union type for condition validation
Condition = Annotated[
    WordMatchCondition
    | MessageCountCondition
    | TimeCondition
    | TokenThresholdCondition
    | CostLimitCondition
    | CallableCondition
    | Jinja2Condition
    | AndCondition
    | OrCondition,
    Field(discriminator="type"),
]
