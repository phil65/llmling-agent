"""Message and token usage models."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel
import tokonomics
from typing_extensions import TypeVar
import yamling

from llmling_agent.common_types import JsonObject, MessageRole  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo  # noqa: TC001


TContent = TypeVar("TContent", str, BaseModel, default=str)
FormatStyle = Literal["simple", "detailed", "markdown", "custom"]
logger = get_logger(__name__)

SIMPLE_TEMPLATE = """{{ name or role.title() }}: {{ content }}"""

DETAILED_TEMPLATE = """From: {{ name or role.title() }}
Time: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
----------------------------------------
{{ content }}
----------------------------------------
{%- if show_costs and cost_info %}
Tokens: {{ "{:,}".format(cost_info.token_usage['total']) }}
Cost: ${{ "%.5f"|format(cost_info.total_cost) }}
{%- if response_time %}
Response time: {{ "%.2f"|format(response_time) }}s
{%- endif %}
{%- endif %}
{%- if show_metadata and metadata %}
Metadata:
{%- for key, value in metadata.items() %}
  {{ key }}: {{ value }}
{%- endfor %}
{%- endif %}
{%- if forwarded_from %}
Forwarded via: {{ forwarded_from|join(' -> ') }}
{%- endif %}"""

MARKDOWN_TEMPLATE = """## {{ name or role.title() }}
*{{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}*

{{ content }}

{%- if show_costs and cost_info %}
---
**Stats:**
- Tokens: {{ "{:,}".format(cost_info.token_usage['total']) }}
- Cost: ${{ "%.4f"|format(cost_info.total_cost) }}
{%- if response_time %}
- Response time: {{ "%.2f"|format(response_time) }}s
{%- endif %}
{%- endif %}

{%- if show_metadata and metadata %}
**Metadata:**
```
{{ metadata|to_yaml }}
```
{%- endif %}

{%- if forwarded_from %}
*Forwarded via: {{ forwarded_from|join(' → ') }}*
{%- endif %}"""

MESSAGE_TEMPLATES = {
    "simple": SIMPLE_TEMPLATE,
    "detailed": DETAILED_TEMPLATE,
    "markdown": MARKDOWN_TEMPLATE,
}


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


@dataclass(frozen=True)
class TokenCost:
    """Combined token and cost tracking."""

    token_usage: TokenUsage
    """Token counts for prompt and completion"""
    total_cost: float
    """Total cost in USD"""

    @classmethod
    async def from_usage(
        cls,
        usage: tokonomics.Usage | None,
        model: str,
        prompt: str,
        completion: str,
    ) -> TokenCost | None:
        """Create result from usage data.

        Args:
            usage: Token counts from model response
            model: Name of the model used
            prompt: The prompt text sent to model
            completion: The completion text received

        Returns:
            TokenCost if usage data available, None otherwise
        """
        if not (
            usage
            and usage.total_tokens is not None
            and usage.request_tokens is not None
            and usage.response_tokens is not None
        ):
            logger.debug("Missing token counts in Usage object")
            return None

        token_usage = TokenUsage(
            total=usage.total_tokens,
            prompt=usage.request_tokens,
            completion=usage.response_tokens,
        )
        logger.debug("Token usage: %s", token_usage)

        cost = await tokonomics.calculate_token_cost(
            model,
            usage.request_tokens,
            usage.response_tokens,
        )
        total_cost = cost.total_cost if cost else 0.0

        return cls(token_usage=token_usage, total_cost=total_cost)


@dataclass
class ChatMessage[TContent]:
    """Common message format for all UI types.

    Generically typed with: ChatMessage[Type of Content]
    The type can either be str or a BaseModel subclass.
    """

    content: TContent
    """Message content, typed as TContent (either str or BaseModel)."""

    role: MessageRole
    """Role of the message sender (user/assistant/system)."""

    model: str | None = None
    """Name of the model that generated this message."""

    metadata: JsonObject = field(default_factory=dict)
    """Additional metadata about the message."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this message was created."""

    cost_info: TokenCost | None = None
    """Token usage and costs for this specific message if available."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this message."""

    response_time: float | None = None
    """Time it took the LLM to respond."""

    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    """List of tool calls made during message generation."""

    name: str | None = None
    """Display name for the message sender in UI."""

    forwarded_from: list[str] = field(default_factory=list)
    """List of agent names (the chain) that forwarded this message to the sender."""

    def to_text_message(self) -> ChatMessage[str]:
        """Convert this message to a text-only version."""
        return dataclasses.replace(self, content=str(self.content))  # type: ignore

    def _get_content_str(self) -> str:
        """Get string representation of content."""
        match self.content:
            case str():
                return self.content
            case BaseModel():
                return self.content.model_dump_json(indent=2)
            case _:
                msg = f"Unexpected content type: {type(self.content)}"
                raise ValueError(msg)

    def to_gradio_format(self) -> tuple[str | None, str | None]:
        """Convert to Gradio chatbot format."""
        content_str = self._get_content_str()
        match self.role:
            case "user":
                return (content_str, None)
            case "assistant":
                return (None, content_str)
            case "system":
                return (None, f"System: {content_str}")

    @property
    def data(self) -> TContent:
        """Get content as typed data. Provides compat to RunResult."""
        return self.content

    def format(
        self,
        style: FormatStyle = "simple",
        *,
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        show_metadata: bool = False,
        show_costs: bool = False,
    ) -> str:
        """Format message with configurable style.

        Args:
            style: Predefined style or "custom" for custom template
            template: Custom Jinja template (required if style="custom")
            variables: Additional variables for template rendering
            show_metadata: Whether to include metadata
            show_costs: Whether to include cost information

        Raises:
            ValueError: If style is "custom" but no template provided
                    or if style is invalid
        """
        from jinja2 import Environment

        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["to_yaml"] = yamling.dump_yaml

        match style:
            case "custom":
                if not template:
                    msg = "Custom style requires a template"
                    raise ValueError(msg)
                template_str = template
            case _ if style in MESSAGE_TEMPLATES:
                template_str = MESSAGE_TEMPLATES[style]
            case _:
                msg = f"Invalid style: {style}"
                raise ValueError(msg)

        template_obj = env.from_string(template_str)
        render_vars = {
            **asdict(self),
            "show_metadata": show_metadata,
            "show_costs": show_costs,
        }
        if variables:
            render_vars.update(variables)

        return template_obj.render(**render_vars)


@dataclass
class Response[TContent]:
    """Response from any source in the agent system."""

    content: ChatMessage[TContent] | str
    """The actual response content (either a ChatMessage or raw text)."""

    source: str
    """Identifies where this response came from (agent/command/tool/stream)."""

    agent_name: str
    """Name of the agent that generated or handled this response."""

    timing: float | None = None
    """Time taken to generate this response in seconds."""

    error: str | None = None
    """Error message if the response generation failed."""

    @property
    def success(self) -> bool:
        """Whether the response was generated successfully."""
        return self.error is None

    @property
    def data(self) -> TContent | str:
        """Direct access to the response content data."""
        return (
            self.content.content
            if isinstance(self.content, ChatMessage)
            else self.content
        )

    def format(
        self,
        style: Literal["simple", "detailed", "markdown"] = "simple",
        *,
        include_context: bool = False,
        **kwargs: Any,
    ) -> str:
        """Format response as string with optional context info."""
        # Get base message formatting
        msg = (
            self.content.format(style, **kwargs)
            if isinstance(self.content, ChatMessage)
            else str(self.content)
        )

        if not include_context:
            return msg

        context_parts = []
        if self.error:
            context_parts.append(f"Error: {self.error}")
        else:
            context_parts.append(f"Source: {self.source}")
            if self.timing:
                context_parts.append(f"Duration: {self.timing:.2f}s")

        return f"{' | '.join(context_parts)}\n{msg}"


@dataclass
class AgentResponse[TResult]:
    """Result from an agent's execution."""

    # TODO: replace with Response

    agent_name: str
    """Name of the agent that produced this result"""

    message: ChatMessage[TResult] | None
    """The actual message with content and metadata"""

    timing: float | None = None
    """Time taken by this agent in seconds"""

    error: str | None = None
    """Error message if agent failed"""

    @property
    def success(self) -> bool:
        """Whether the agent completed successfully."""
        return self.error is None

    @property
    def response(self) -> TResult | None:
        """Convenient access to message content."""
        return self.message.content if self.message else None


class TeamResponse(list[AgentResponse[Any]]):
    """Results from a team execution."""

    def __init__(
        self, responses: list[AgentResponse[Any]], start_time: datetime | None = None
    ):
        super().__init__(responses)
        self.start_time = start_time or datetime.now()
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def successful(self) -> list[AgentResponse[Any]]:
        """Get only successful responses."""
        return [r for r in self if r.success]

    @property
    def failed(self) -> list[AgentResponse[Any]]:
        """Get failed responses."""
        return [r for r in self if not r.success]

    def by_agent(self, name: str) -> AgentResponse[Any] | None:
        """Get response from specific agent."""
        return next((r for r in self if r.agent_name == name), None)

    def format_durations(self) -> str:
        """Format execution times."""
        parts = [f"{r.agent_name}: {r.timing:.2f}s" for r in self if r.timing is not None]
        return f"Individual times: {', '.join(parts)}\nTotal time: {self.duration:.2f}s"

    def to_chat_message(self) -> ChatMessage[str]:
        """Convert team response to a single chat message."""
        # Combine all responses into one structured message
        content = "\n\n".join(
            f"[{response.agent_name}]: {response.message.content}"
            for response in self
            if response.message
        )
        meta = {
            "type": "team_response",
            "agents": [r.agent_name for r in self],
            "duration": self.duration,
            "success_count": len(self.successful),
        }
        return ChatMessage(content=content, role="assistant", metadata=meta)  # type: ignore
