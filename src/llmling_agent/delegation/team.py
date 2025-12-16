"""Parallel, unordered group of agents / nodes."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from anyenv.async_run import as_generated
from toprompt import to_prompt

from llmling_agent.common_types import SupportsRunStream
from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.log import get_logger
from llmling_agent.messaging import AgentResponse, ChatMessage, TeamResponse
from llmling_agent.messaging.processing import finalize_message, prepare_prompts
from llmling_agent.utils.now import get_now


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from toprompt import AnyPromptType

    from llmling_agent import MessageNode
    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import PromptCompatible
    from llmling_agent.talk import Talk
    from llmling_agent_config.task import Job


async def normalize_stream_for_teams(
    node: MessageNode[Any, Any],
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[tuple[MessageNode[Any, Any], RichAgentStreamEvent[Any]]]:
    """Normalize any streaming node to yield (node, event) tuples for team composition.

    Args:
        node: The streaming node (Agent, Team, etc.)
        *args: Arguments to pass to run_stream
        **kwargs: Keyword arguments to pass to run_stream

    Yields:
        Tuples of (node, event) where node is the MessageNode instance
        and event is the streaming event from that node.
    """
    if not isinstance(node, SupportsRunStream):
        msg = f"Node {node.name} does not support streaming"
        raise TypeError(msg)

    stream = node.run_stream(*args, **kwargs)
    async for item in stream:
        if isinstance(item, tuple):
            # Already normalized (from Team or other composite node)
            yield item
        else:
            # Raw event (from Agent) - wrap it with the source node
            yield (node, item)


class Team[TDeps = None](BaseTeam[TDeps, Any]):
    """Group of agents that can execute together."""

    async def execute(self, *prompts: PromptCompatible | None, **kwargs: Any) -> TeamResponse:
        """Run all agents in parallel with monitoring."""
        from llmling_agent.talk.talk import Talk

        self._team_talk.clear()
        start_time = get_now()
        responses: list[AgentResponse[Any]] = []
        errors: dict[str, Exception] = {}
        final_prompt = list(prompts)
        if self.shared_prompt:
            final_prompt.insert(0, self.shared_prompt)
        combined_prompt = "\n".join([await to_prompt(p) for p in final_prompt])
        all_nodes = list(await self.pick_agents(combined_prompt))
        # Create Talk connections for monitoring this execution
        execution_talks: list[Talk[Any]] = []
        for node in all_nodes:
            # No actual forwarding, just for tracking
            talk = Talk[Any](node, [], connection_type="run", queued=True, queue_strategy="latest")
            execution_talks.append(talk)
            self._team_talk.append(talk)  # Add to base class's TeamTalk

        async def _run(node: MessageNode[TDeps, Any]) -> None:
            try:
                start = perf_counter()
                message = await node.run(*final_prompt, **kwargs)
                timing = perf_counter() - start
                r = AgentResponse(agent_name=node.name, message=message, timing=timing)
                responses.append(r)
                # Update talk stats for this agent
                talk = next(t for t in execution_talks if t.source == node)
                talk._stats.messages.append(message)
            except Exception as e:  # noqa: BLE001
                errors[node.name] = e

        # Run all agents in parallel
        await asyncio.gather(*[_run(node) for node in all_nodes])
        return TeamResponse(responses=responses, start_time=start_time, errors=errors)

    def __prompt__(self) -> str:
        """Format team info for prompts."""
        members = ", ".join(a.name for a in self.nodes)
        desc = f" - {self.description}" if self.description else ""
        return f"Parallel Team {self.name!r}{desc}\nMembers: {members}"

    async def run_iter(
        self,
        *prompts: AnyPromptType,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages as they arrive from parallel execution."""
        queue: asyncio.Queue[ChatMessage[Any] | None] = asyncio.Queue()
        failures: dict[str, Exception] = {}

        async def _run(node: MessageNode[TDeps, Any]) -> None:
            try:
                message = await node.run(*prompts, **kwargs)
                await queue.put(message)
            except Exception as e:
                logger.exception("Error executing node", name=node.name)
                failures[node.name] = e
                # Put None to maintain queue count
                await queue.put(None)

        # Get nodes to run
        combined_prompt = "\n".join([await to_prompt(p) for p in prompts])
        all_nodes = list(await self.pick_agents(combined_prompt))

        # Start all agents
        tasks = [asyncio.create_task(_run(n), name=f"run_{n.name}") for n in all_nodes]

        try:
            # Yield messages as they arrive
            for _ in all_nodes:
                if msg := await queue.get():
                    yield msg

            # If any failures occurred, raise error with details
            if failures:
                error_details = "\n".join(f"- {name}: {error}" for name, error in failures.items())
                error_msg = f"Some nodes failed to execute:\n{error_details}"
                raise RuntimeError(error_msg)

        finally:
            # Clean up any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def run(
        self,
        *prompts: PromptCompatible | None,
        wait_for_connections: bool | None = None,
        store_history: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[list[Any]]:
        """Run all agents in parallel and return combined message."""
        # Prepare prompts and create user message
        user_msg, processed_prompts, original_message = await prepare_prompts(*prompts)
        self.message_received.emit(user_msg)

        # Execute team logic
        result = await self.execute(*processed_prompts, **kwargs)
        message_id = str(uuid4())  # Always generate unique response ID
        message = ChatMessage(
            content=[r.message.content for r in result if r.message],
            messages=[m for r in result if r.message for m in r.message.messages],
            role="assistant",
            name=self.name,
            message_id=message_id,
            conversation_id=user_msg.conversation_id,
            metadata={
                "agent_names": [r.agent_name for r in result],
                "errors": {name: str(error) for name, error in result.errors.items()},
                "start_time": result.start_time.isoformat(),
            },
        )

        # Teams typically don't store history by default, but allow it
        if store_history:
            # Teams could implement their own history management here if needed
            pass

        # Finalize and route message
        return await finalize_message(
            message,
            user_msg,
            self,
            self.connections,
            original_message,
            wait_for_connections,
        )

    async def run_stream(
        self,
        *prompts: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[MessageNode[Any, Any], RichAgentStreamEvent[Any]]]:
        """Stream responses from all team members in parallel.

        Args:
            prompts: Input prompts to process in parallel
            kwargs: Additional arguments passed to each agent

        Yields:
            Tuples of (agent, event) where agent is the Agent instance
            and event is the streaming event from that agent.
        """
        # Get nodes to run
        combined_prompt = "\n".join([await to_prompt(p) for p in prompts])
        all_nodes = list(await self.pick_agents(combined_prompt))

        # Create list of streams that yield (agent, event) tuples
        agent_streams = [
            normalize_stream_for_teams(agent, *prompts, **kwargs)
            for agent in all_nodes
            if isinstance(agent, SupportsRunStream)
        ]

        # Merge all agent streams
        async for agent_event_tuple in as_generated(agent_streams):
            yield agent_event_tuple

    async def run_job[TJobResult](
        self,
        job: Job[TDeps, TJobResult],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> list[AgentResponse[TJobResult]]:
        """Execute a job across all team members in parallel.

        Args:
            job: Job configuration to execute
            store_history: Whether to add job execution to conversation history
            include_agent_tools: Whether to include agent's tools alongside job tools

        Returns:
            List of responses from all agents

        Raises:
            JobError: If job execution fails for any agent
            ValueError: If job configuration is invalid
        """
        from llmling_agent import Agent
        from llmling_agent.tasks import JobError

        responses: list[AgentResponse[TJobResult]] = []
        errors: dict[str, Exception] = {}
        start_time = get_now()

        # Validate dependencies for all agents
        if job.required_dependency is not None:
            invalid_agents = [
                agent.name
                for agent in self.iter_agents()
                if not isinstance(agent.context.data, job.required_dependency)
            ]
            if invalid_agents:
                msg = (
                    f"Agents {', '.join(invalid_agents)} don't have required "
                    f"dependency type: {job.required_dependency}"
                )
                raise JobError(msg)

        try:
            # Load knowledge for all agents if provided
            if job.knowledge:
                # TODO: resources
                tools = [t.name for t in job.get_tools()]
                await self.distribute(content="", tools=tools)

            prompt = await job.get_prompt()

            async def _run(agent: MessageNode[TDeps, TJobResult]) -> None:
                assert isinstance(agent, Agent)
                try:
                    async with agent.tools.temporary_tools(
                        job.get_tools(), exclusive=not include_agent_tools
                    ):
                        start = perf_counter()
                        resp = AgentResponse(
                            agent_name=agent.name,
                            message=await agent.run(prompt, store_history=store_history),
                            timing=perf_counter() - start,
                        )
                        responses.append(resp)
                except Exception as e:  # noqa: BLE001
                    errors[agent.name] = e

            # Run job in parallel on all agents
            await asyncio.gather(*[_run(node) for node in self.nodes])
            return TeamResponse(responses=responses, start_time=start_time, errors=errors)

        except Exception as e:
            msg = "Job execution failed"
            logger.exception(msg)
            raise JobError(msg) from e


if __name__ == "__main__":

    async def main() -> None:
        from llmling_agent import Agent, TeamRun

        agent_a = Agent(name="A", model="test")
        agent_b = Agent(name="B", model="test")
        agent_c = Agent(name="C", model="test")
        # Test Team containing TeamRun (parallel containing sequential)
        inner_run = TeamRun([agent_a, agent_b], name="Sequential")
        outer_team = Team([inner_run, agent_c], name="Parallel")

        print("Testing Team containing TeamRun...")
        async for node, event in outer_team.run_stream("test"):
            print(f"{node.name}: {type(event).__name__}")

    asyncio.run(main())
