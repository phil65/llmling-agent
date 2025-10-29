"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.messaging.messagenode import MessageNode


if TYPE_CHECKING:
    from llmling.config.models import Resource

    from llmling_agent.agent.agent import Agent
    from llmling_agent.delegation.base_team import BaseTeam
    from llmling_agent.delegation.team import Team
    from llmling_agent.delegation.teamrun import TeamRun
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent_config.output_types import StructuredResponseConfig
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
        output_type: type[TResult] | str | StructuredResponseConfig,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Initialize structured agent wrapper.

        Args:
            agent: Base agent to wrap
            output_type: Expected result type:
                - BaseModel / dataclasses
                - Name of response definition in manifest
                - Complete response definition instance
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description

        Raises:
            ValueError: If named response type not found in manifest
        """
        from llmling_agent.agent.agent import Agent

        logger.debug("StructuredAgent.run output_type = %s", output_type)
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

    def __and__(
        self, other: Agent[Any, Any] | Team[Any] | ProcessorCallback[TResult]
    ) -> Team[TDeps]:
        return self._agent.__and__(other)

    def __or__(self, other: Agent[Any, Any] | ProcessorCallback | BaseTeam) -> TeamRun:
        return self._agent.__or__(other)

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
        if job.required_return_type != self._output_type:
            msg = (
                f"Agent result type ({self._output_type}) "
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

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self._pending_tasks or self._background_task)
