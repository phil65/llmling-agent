"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai import ModelResponse, _agent_graph
from pydantic_ai.result import FinalResult
from pydantic_graph import End

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai import ModelMessage

    from llmling_agent.messaging.messages import TokenCost
    from llmling_agent.tools import ToolCallInfo


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
