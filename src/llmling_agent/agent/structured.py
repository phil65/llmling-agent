"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, get_type_hints, overload

from pydantic import ValidationError

from llmling_agent.log import get_logger
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.utils.result_utils import to_type
from llmling_agent_config.result_types import StructuredResponseConfig


if TYPE_CHECKING:
    from llmling.config.models import Resource

    from llmling_agent.agent.agent import Agent
    from llmling_agent.delegation.base_team import BaseTeam
    from llmling_agent.delegation.team import Team
    from llmling_agent.delegation.teamrun import TeamRun
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.talk.stats import MessageStats
    from llmling_agent_config.task import Job
    from llmling_agent_providers.callback import ProcessorCallback


logger = get_logger(__name__)


class StructuredAgent[TDeps = None, TResult = str](MessageNode[TDeps, TResult]):
    """Wrapper for Agent that enforces a specific result type.

    This wrapper ensures the agent always returns results of the specified type.
    The type can be provided as:
    - A Python type for validation
    - A response definition name from the manifest
    - A complete response definition instance
    """

    def __init__(
        self,
        agent: Agent[TDeps] | StructuredAgent[TDeps, TResult] | Callable[..., TResult],
        result_type: type[TResult] | str | StructuredResponseConfig,
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
        from llmling_agent.agent.agent import Agent

        logger.debug("StructuredAgent.run result_type = %s", result_type)
        match agent:
            case StructuredAgent():
                self._agent: Agent[TDeps] = agent._agent
            case Callable():
                self._agent = Agent[TDeps](provider=agent, name=agent.__name__)
            case Agent():
                self._agent = agent
            case _:
                msg = "Invalid agent type"
                raise ValueError(msg)

        super().__init__(name=self._agent.name)

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
            case StructuredResponseConfig():
                # For response definitions, use as-is
                # (overrides don't apply to complete definitions)
                self._agent.set_result_type(result_type)

    def __and__(
        self, other: Agent[Any, Any] | Team[Any] | ProcessorCallback[TResult]
    ) -> Team[TDeps]:
        return self._agent.__and__(other)

    def __or__(self, other: Agent[Any, Any] | ProcessorCallback | BaseTeam) -> TeamRun:
        return self._agent.__or__(other)

    async def validate_against(
        self,
        prompt: str,
        criteria: type[TResult],
        **kwargs: Any,
    ) -> bool:
        """Check if agent's response satisfies stricter criteria."""
        result = await self.run(prompt, **kwargs)
        try:
            criteria.model_validate(result.content.model_dump())  # type: ignore
        except ValidationError:
            return False
        else:
            return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

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
        result_type: type[TNewResult] | str | StructuredResponseConfig,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TNewResult]: ...

    def to_structured[TNewResult](
        self,
        result_type: type[TNewResult] | str | StructuredResponseConfig | None,
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

    async def get_stats(self) -> MessageStats:
        return await self._agent.get_stats()

    async def run_job(
        self,
        job: Job[TDeps, TResult],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> ChatMessage[TResult]:
        """Execute a pre-defined job ensuring type compatibility.

        Args:
            job: Job configuration to execute
            store_history: Whether to add job execution to conversation history
            include_agent_tools: Whether to include agent's tools alongside job tools

        Returns:
            Task execution result

        Raises:
            JobError: If job execution fails or types don't match
            ValueError: If job configuration is invalid
        """
        from llmling_agent.tasks import JobError

        # Validate dependency requirement
        if job.required_dependency is not None:  # noqa: SIM102
            if not isinstance(self.context.data, job.required_dependency):
                msg = (
                    f"Agent dependencies ({type(self.context.data)}) "
                    f"don't match job requirement ({job.required_dependency})"
                )
                raise JobError(msg)

        # Validate return type requirement
        if job.required_return_type != self._result_type:
            msg = (
                f"Agent result type ({self._result_type}) "
                f"doesn't match job requirement ({job.required_return_type})"
            )
            raise JobError(msg)

        # Load task knowledge if provided
        if job.knowledge:
            # Add knowledge sources to context
            resources: list[Resource | str] = list(job.knowledge.paths) + list(
                job.knowledge.resources
            )
            for source in resources:
                await self.conversation.load_context_source(source)
            for prompt in job.knowledge.prompts:
                await self.conversation.load_context_source(prompt)

        try:
            # Register task tools temporarily
            tools = job.get_tools()

            # Use temporary tools
            with self._agent.tools.temporary_tools(
                tools, exclusive=not include_agent_tools
            ):
                # Execute job using StructuredAgent's run to maintain type safety
                return await self.run(await job.get_prompt(), store_history=store_history)

        except Exception as e:
            msg = f"Task execution failed: {e}"
            logger.exception(msg)
            raise JobError(msg) from e

    @classmethod
    def from_callback(
        cls,
        callback: ProcessorCallback[TResult],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> StructuredAgent[None, TResult]:
        """Create a structured agent from a processing callback.

        Args:
            callback: Function to process messages. Can be:
                - sync or async
                - with or without context
                - with explicit return type
            name: Optional name for the agent
            **kwargs: Additional arguments for agent

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

        name = name or callback.__name__ or "processor"
        provider = CallbackProvider(callback, name=name)
        agent = Agent[None, Any](provider=provider, name=name, **kwargs)
        # Get return type from signature for validation
        hints = get_type_hints(callback)
        return_type = hints.get("return")

        # If async, unwrap from Awaitable
        if (
            return_type
            and hasattr(return_type, "__origin__")
            and return_type.__origin__ is Awaitable
        ):
            return_type = return_type.__args__[0]
        return StructuredAgent[None, TResult](agent, return_type or str)  # type: ignore

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self._pending_tasks or self._background_task)
