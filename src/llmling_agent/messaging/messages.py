"""Message and token usage models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar
import uuid
from uuid import uuid4

from genai_prices import calc_price
from pydantic import BaseModel
from pydantic_ai import (
    BaseToolReturnPart,
    FilePart,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    TextPart,
    ToolCallPart,
    UserContent,
    UserPromptPart,
)
import tokonomics

from llmling_agent.common_types import MessageRole, SimpleJsonType  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.utils.now import get_now


if TYPE_CHECKING:
    from datetime import datetime

    from pydantic_ai import (
        FinishReason,
        ModelMessage,
        ModelRequestPart,
        ModelResponsePart,
        RunUsage,
    )

    from llmling_agent.tools import ToolCallInfo


TContent = TypeVar("TContent", str, BaseModel)
FormatStyle = Literal["simple", "detailed", "markdown", "custom"]
logger = get_logger(__name__)

SIMPLE_TEMPLATE = """{{ name or role.title() }}: {{ content }}"""

DETAILED_TEMPLATE = """From: {{ name or role.title() }}
Time: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
----------------------------------------
{{ content }}
----------------------------------------
{%- if show_costs and cost_info %}
Tokens: {{ "{:,}".format(cost_info.token_usage.total_tokens) }}
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
- Tokens: {{ "{:,}".format(cost_info.token_usage.total_tokens) }}
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

{% if forwarded_from %}

*Forwarded via: {{ forwarded_from|join(' → ') }}*
{% endif %}"""

MESSAGE_TEMPLATES = {
    "simple": SIMPLE_TEMPLATE,
    "detailed": DETAILED_TEMPLATE,
    "markdown": MARKDOWN_TEMPLATE,
}


@dataclass(frozen=True)
class TokenCost:
    """Combined token and cost tracking."""

    token_usage: RunUsage
    """Token counts for prompt and completion"""
    total_cost: Decimal
    """Total cost in USD"""

    @classmethod
    async def from_usage(cls, usage: RunUsage | None, model: str) -> TokenCost | None:
        """Create result from usage data.

        Args:
            usage: Token counts from model response
            model: Name of the model used


        Returns:
            TokenCost if usage data available, None otherwise
        """
        if not (
            usage and usage.input_tokens is not None and usage.output_tokens is not None
        ):
            logger.debug("Missing token counts in Usage object")
            return None
        logger.debug("Token usage", usage=usage)

        # return cls(token_usage=token_usage, total_cost=Decimal(total_cost))
        if model in {"None", "test"}:
            price = Decimal(0)
        else:
            parts = model.split(":", 1)
            try:
                price_data = calc_price(
                    usage,
                    model_ref=parts[1] if len(parts) > 1 else parts[0],
                    provider_id=parts[0] if len(parts) > 1 else "openai",
                )
                price = price_data.total_price
            except Exception:  # noqa: BLE001
                cost = await tokonomics.calculate_token_cost(
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                )
                price = Decimal(cost.total_cost if cost else 0)

        return cls(token_usage=usage, total_cost=price)


@dataclass
class ChatMessage[TContent]:
    """Common message format for all UI types.

    Generically typed with: ChatMessage[Type of Content]
    The type can either be str or a BaseModel subclass.
    """

    content: TContent
    """Message content, typed as TContent (either str or BaseModel)."""

    role: MessageRole
    """Role of the message sender (user/assistant)."""

    metadata: SimpleJsonType = field(default_factory=dict)
    """Additional metadata about the message."""

    timestamp: datetime = field(default_factory=get_now)
    """When this message was created."""

    cost_info: TokenCost | None = None
    """Token usage and costs for this specific message if available."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this message."""

    conversation_id: str | None = None
    """ID of the conversation this message belongs to."""

    response_time: float | None = None
    """Time it took the LLM to respond."""

    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    """List of tool calls made during message generation."""

    associated_messages: list[ChatMessage[Any]] = field(default_factory=list)
    """List of messages which were generated during the the creation of this messsage."""

    name: str | None = None
    """Display name for the message sender in UI."""

    forwarded_from: list[str] = field(default_factory=list)
    """List of agent names (the chain) that forwarded this message to the sender."""

    provider_details: dict[str, Any] = field(default_factory=dict)
    """Provider specific metadata / extra information."""

    parts: Sequence[ModelResponsePart | ModelRequestPart] = field(default_factory=list)
    """The parts of the model message."""

    messages: list[ModelMessage] = field(default_factory=list)
    """List of messages which were generated during the the creation of this messsage."""

    usage: RequestUsage = field(default_factory=RequestUsage)
    """Usage information for the request.

    This has a default to make tests easier,
    and to support loading old messages where usage will be missing.
    """

    model_name: str | None = None
    """The name of the model that generated the response."""

    provider_name: str | None = None
    """The name of the LLM provider that generated the response."""

    provider_response_id: str | None = None
    """request ID as specified by the model provider.

    This can be used to track the specific request to the model."""

    finish_reason: FinishReason | None = None
    """Reason the model finished generating the response.

    Normalized to OpenTelemetry values."""

    @property
    def last_message(self) -> ModelMessage:
        """Return the last message from the message history."""
        # The response may not be the very last item if it contained an output tool call.
        # See `CallToolsNode._handle_final_result`.
        for message in reversed(self.messages):
            if isinstance(message, ModelMessage):
                return message
        msg = "No last message found in the message history"
        raise ValueError(msg)  # pragma: no cover

    # @property
    # def parts(self) -> Sequence[ModelRequestPart] | Sequence[ModelResponsePart]:
    #     """The parts of the last model message."""
    #     return self.last_message.parts

    @property
    def kind(self) -> Literal["request", "response"]:
        """Role of the message."""
        match self.role:
            case "assistant":
                return "response"
            case "user":
                return "request"

    def to_pydantic_ai(self) -> Sequence[ModelMessage]:
        """Convert this message to a Pydantic model."""
        if self.messages:
            return self.messages
        match self.kind:
            case "request":
                return [ModelRequest(parts=self.parts, instructions=None)]  # type: ignore
            case "response":
                return [
                    ModelResponse(
                        parts=self.parts,  # type: ignore
                        usage=self.usage,
                        model_name=self.model_name,
                        timestamp=self.timestamp,
                        provider_name=self.provider_name,
                        provider_details=self.provider_details,
                        finish_reason=self.finish_reason,
                        provider_response_id=self.provider_response_id,
                    )
                ]

    @classmethod
    def user_prompt[TPromptContent: str | Sequence[UserContent] = str](
        cls,
        message: TPromptContent,
        conversation_id: str | None = None,
    ) -> ChatMessage[TPromptContent]:
        """Create a user prompt message."""
        return ChatMessage(
            messages=[ModelRequest(parts=[UserPromptPart(content=message)])],
            role="user",
            content=message,
            conversation_id=conversation_id or str(uuid4()),
        )

    @classmethod
    def from_pydantic_ai[TContentType](
        cls,
        content: TContentType,
        message: ModelMessage,
        conversation_id: str | None = None,
        name: str | None = None,
        message_id: str | None = None,
        forwarded_from: list[str] | None = None,
    ) -> ChatMessage[TContentType]:
        """Convert a Pydantic model to a ChatMessage."""
        match message:
            case ModelRequest(parts=parts, instructions=_instructions):
                return ChatMessage(
                    parts=parts,
                    content=content,
                    role="user" if message.kind == "request" else "assistant",
                    message_id=message_id or str(uuid.uuid4()),
                    # instructions=instructions,
                    forwarded_from=forwarded_from or [],
                    name=name,
                )
            case ModelResponse(
                parts=parts,
                usage=usage,
                model_name=model_name,
                timestamp=timestamp,
                provider_name=provider_name,
                provider_details=provider_details,
                finish_reason=finish_reason,
                provider_response_id=provider_response_id,
            ):
                return ChatMessage(
                    role="user" if message.kind == "request" else "assistant",
                    content=content,
                    parts=parts,
                    usage=usage,
                    message_id=message_id or str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    model_name=model_name,
                    timestamp=timestamp,
                    provider_name=provider_name,
                    provider_details=provider_details or {},
                    finish_reason=finish_reason,
                    provider_response_id=provider_response_id,
                    name=name,
                    forwarded_from=forwarded_from or [],
                )
            case _:
                msg = f"Unknown message kind: {message.kind}"
                raise ValueError(msg)

    def forwarded(self, previous_message: ChatMessage[Any]) -> Self:
        """Create new message showing it was forwarded from another message.

        Args:
            previous_message: The message that led to this one's creation

        Returns:
            New message with updated chain showing the path through previous message
        """
        from_ = [*previous_message.forwarded_from, previous_message.name or "unknown"]
        return replace(self, forwarded_from=from_)

    @property
    def response(self) -> ModelResponse:
        """Return the last response from the message history."""
        # The response may not be the very last item if it contained an output tool call.
        #  See `CallToolsNode._handle_final_result`.
        for message in reversed(self.messages):
            if isinstance(message, ModelResponse):
                return message
        msg = "No response found in the message history"
        raise ValueError(msg)

    def to_request(self) -> Self:
        """Convert this message to a request message.

        If the message is already a request (user role), this is a no-op.
        If it's a response (assistant role), converts response parts to user content.

        Returns:
            New ChatMessage with role='user' and converted parts
        """
        if self.role == "user":
            # Already a request, return as-is
            return self

        # Convert response parts to user content
        converted_parts: list[Any] = []
        user_content: list[UserContent] = []

        for part in self.parts:
            match part:
                case TextPart(content=text_content):
                    # Text parts become user content strings
                    user_content.append(text_content)
                case FilePart(content=binary_content):
                    # File parts (images, etc.) become user content directly
                    user_content.append(binary_content)
                case BaseToolReturnPart(
                    content=str() | Sequence(UserContent()) as content
                ):
                    user_content.append(content)
                case BaseToolReturnPart():
                    # Tool return parts become user content strings
                    user_content.append(part.model_response_str())
                case ToolCallPart():
                    # Tool return parts become user content strings
                    user_content.append(part.args_as_json_str())
                case _:
                    pass

        # Create new UserPromptPart with converted content
        if user_content:
            converted_parts = [UserPromptPart(content=user_content)]
        else:
            converted_parts = [UserPromptPart(content=str(self.content))]

        return replace(self, role="user", parts=converted_parts, cost_info=None)

    @property
    def data(self) -> TContent:
        """Get content as typed data. Provides compat to AgentRunResult."""
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
        from jinjarope import Environment
        import yamling

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
        vars_ = {
            **(self.__dict__),
            "show_metadata": show_metadata,
            "show_costs": show_costs,
        }
        print(vars_)
        if variables:
            vars_.update(variables)

        return template_obj.render(**vars_)


@dataclass
class AgentResponse[TResult]:
    """Result from an agent's execution."""

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


class TeamResponse[TMessageContent](list[AgentResponse[Any]]):
    """Results from a team execution."""

    def __init__(
        self,
        responses: list[AgentResponse[TMessageContent]],
        start_time: datetime | None = None,
        errors: dict[str, Exception] | None = None,
    ):
        super().__init__(responses)
        self.start_time = start_time or get_now()
        self.end_time = get_now()
        self.errors = errors or {}

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success(self) -> bool:
        """Whether all agents completed successfully."""
        return not bool(self.errors)

    @property
    def failed_agents(self) -> list[str]:
        """Names of agents that failed."""
        return list(self.errors.keys())

    def by_agent(self, name: str) -> AgentResponse[TMessageContent] | None:
        """Get response from specific agent."""
        return next((r for r in self if r.agent_name == name), None)

    def format_durations(self) -> str:
        """Format execution times."""
        parts = [f"{r.agent_name}: {r.timing:.2f}s" for r in self if r.timing is not None]
        return f"Individual times: {', '.join(parts)}\nTotal time: {self.duration:.2f}s"

    # TODO: could keep TResultContent for len(messages) == 1
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
            "success_count": len(self),
        }
        return ChatMessage(content=content, role="assistant", metadata=meta)  # type: ignore
