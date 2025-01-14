"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, get_type_hints, overload

from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.responses.models import BaseResponseDefinition, ResponseDefinition
from llmling_agent.responses.utils import to_type


if TYPE_CHECKING:
    from types import TracebackType

    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.agent.agent import Agent
    from llmling_agent.common_types import ModelType
    from llmling_agent.models.context import AgentContext
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.tools.manager import ToolManager
    from llmling_agent_providers.callback import ProcessorCallback


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
TDeps = TypeVar("TDeps", default=Any)


class StructuredAgent[TDeps, TResult]:
    """Wrapper for Agent that enforces a specific result type.

    This wrapper ensures the agent always returns results of the specified type.
    The type can be provided as:
    - A Python type for validation
    - A response definition name from the manifest
    - A complete response definition instance
    """

    def __init__(
        self,
        agent: AnyAgent[TDeps, TResult],
        result_type: type[TResult] | str | ResponseDefinition,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Initialize structured agent wrapper.

        Args:
            agent: Base agent to wrap
            result_type: Expected result type:
                - BaseModel / dataclasses
                - Name of response definition in manifest
                - Complete response definition instance
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description

        Raises:
            ValueError: If named response type not found in manifest
        """
        logger.debug("StructuredAgent.run result_type = %s", result_type)
        if isinstance(agent, StructuredAgent):
            self._agent: Agent[TDeps] = agent._agent
        else:
            self._agent = agent
        self._result_type = to_type(result_type)
        agent.set_result_type(result_type)

        match result_type:
            case type() | str():
                # For types and named definitions, use overrides if provided
                self._agent.set_result_type(
                    result_type,
                    tool_name=tool_name,
                    tool_description=tool_description,
                )
            case BaseResponseDefinition():
                # For response definitions, use as-is
                # (overrides don't apply to complete definitions)
                self._agent.set_result_type(result_type)

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers.

        Called when agent enters its async context. Sets up any configured
        MCP servers and their tools.
        """
        await self._agent.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit async context."""
        await self._agent.__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        *prompt: AnyPromptType | TResult,
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
    ) -> ChatMessage[TResult]:
        """Run with fixed result type.

        Args:
            prompt: Any prompt-compatible object or structured objects of type TResult
            result_type: Expected result type:
                - BaseModel / dataclasses
                - Name of response definition in manifest
                - Complete response definition instance
            deps: Optional dependencies for the agent
            message_history: Optional previous messages for context
            model: Optional model override
            usage: Optional usage tracking
        """
        typ = result_type or self._result_type
        return await self._agent.run(*prompt, result_type=typ, deps=deps, model=model)

    def __repr__(self) -> str:
        type_name = getattr(self._result_type, "__name__", str(self._result_type))
        return f"StructuredAgent({self._agent!r}, result_type={type_name})"

    def __prompt__(self) -> str:
        type_name = getattr(self._result_type, "__name__", str(self._result_type))
        base_info = self._agent.__prompt__()
        return f"{base_info}\nStructured output type: {type_name}"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    @property
    def context(self) -> AgentContext[TDeps]:
        return self._agent.context

    @context.setter
    def context(self, value: Any):
        self._agent.context = value

    @property
    def name(self) -> str:
        return self._agent.name

    @name.setter
    def name(self, value: str):
        self._agent.name = value

    @property
    def tools(self) -> ToolManager:
        return self._agent.tools

    @overload
    def to_structured(
        self,
        result_type: None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Agent[TDeps]: ...

    @overload
    def to_structured[TNewResult](
        self,
        result_type: type[TNewResult] | str | ResponseDefinition,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TNewResult]: ...

    def to_structured[TNewResult](
        self,
        result_type: type[TNewResult] | str | ResponseDefinition | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Agent[TDeps] | StructuredAgent[TDeps, TNewResult]:
        if result_type is None:
            return self._agent

        return StructuredAgent(
            self._agent,
            result_type=result_type,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    @classmethod
    def from_callback(
        cls,
        callback: ProcessorCallback[TResult],
        *,
        deps: TDeps | None = None,
        name: str | None = None,
        debug: bool = False,
    ) -> StructuredAgent[TDeps, TResult]:
        """Create a structured agent from a processing callback.

        Args:
            callback: Function to process messages. Can be:
                - sync or async
                - with or without context
                - with explicit return type
            deps: Optional dependencies for the agent
            name: Optional name for the agent
            debug: Whether to enable debug mode

        Example:
            ```python
            class AnalysisResult(BaseModel):
                sentiment: float
                topics: list[str]

            def analyze(msg: str) -> AnalysisResult:
                return AnalysisResult(sentiment=0.8, topics=["tech"])

            analyzer = StructuredAgent.from_callback(analyze)
            ```
        """
        from llmling_agent.agent.agent import Agent
        from llmling_agent_providers.callback import CallbackProvider

        provider = CallbackProvider[TDeps](
            callback,
            name=name or callback.__name__ or "processor",
            debug=debug,
        )
        agent = Agent[TDeps](provider=provider)
        if deps is not None:
            agent.context.data = deps

        # Get return type from signature for validation
        hints = get_type_hints(callback)
        return_type = hints.get("return")

        # If async, unwrap from Awaitable
        if return_type and hasattr(return_type, "__origin__"):
            from collections.abc import Awaitable

            if return_type.__origin__ is Awaitable:
                return_type = return_type.__args__[0]
        return cls(agent, return_type or str)  # type: ignore
