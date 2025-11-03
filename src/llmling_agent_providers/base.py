"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
)

from psygnal import Signal
from pydantic_ai import ModelResponse, _agent_graph
from pydantic_ai.result import FinalResult
from pydantic_graph import End

from llmling_agent.log import get_logger
from llmling_agent.tools import ToolCallInfo


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import (
        AgentRunResultEvent,
        AgentStreamEvent,
        ModelMessage,
        UsageLimits,
    )
    from pydantic_ai.models import Model

    from llmling_agent.agent.context import AgentContext
    from llmling_agent.common_types import ModelType
    from llmling_agent.messaging.messages import ChatMessage, TokenCost
    from llmling_agent.models.content import Content
    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)
TResult_co = TypeVar("TResult_co", default=str, covariant=True)
type TNode[TResult_co] = _agent_graph.AgentNode | End[FinalResult[TResult_co]]


@dataclass
class ProviderResponse:
    """Raw response data from provider."""

    content: Any
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    messages: list[ModelMessage] = field(default_factory=list)
    response: ModelResponse = field(default_factory=lambda: ModelResponse(parts=[]))
    cost_and_usage: TokenCost | None = None
    provider_details: dict[str, Any] | None = None


class AgentProvider[TDeps]:
    """Base class for agent providers."""

    tool_used = Signal(ToolCallInfo)
    NAME: str

    def __init__(
        self,
        *,
        name: str = "agent",
        context: AgentContext[TDeps] | None = None,
        debug: bool = False,
    ):
        super().__init__()
        self._name = name
        self._model: str | Model | None = None
        self._context = context
        self._debug = debug

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def set_model(self, model: ModelType):
        """Default no-op implementation for setting model."""

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get context."""
        if self._context is None:
            msg = "Context not set"
            raise RuntimeError(msg)
        return self._context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set context."""
        self._context = value

    @property
    def model_name(self) -> str | None:
        """Get model name."""
        return None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "agent"

    @name.setter
    def name(self, value: str | None):
        """Set agent name."""
        self._name = value or "agent"

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        output_type: type[Any] | None = None,
        model: ModelType = None,
        tools: list[Tool] | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response. Must be implemented by providers."""
        raise NotImplementedError

    def stream_events[TResult](
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        output_type: type[TResult] | None = None,
        model: ModelType = None,
        tools: list[Tool] | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[TResult]]:
        """Stream a response. Must be implemented by providers."""
        raise NotImplementedError

    async def supports_feature(self, capability: Literal["vision"]) -> bool:
        """Check if provider supports a specific capability."""
        import tokonomics

        match capability:
            case "vision":
                if not self.model_name:
                    return False
                caps = await tokonomics.get_model_capabilities(self.model_name)
                return bool(caps and caps.supports_vision)
            case _:
                return False
