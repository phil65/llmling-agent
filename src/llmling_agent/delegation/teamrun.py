"""Sequential, ordered group of agents / nodes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import uuid4

from pydantic_ai import PartDeltaEvent, TextPartDelta

from llmling_agent.common_types import SupportsRunStream
from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.delegation.team import normalize_stream_for_teams
from llmling_agent.log import get_logger
from llmling_agent.messaging import AgentResponse, ChatMessage, TeamResponse
from llmling_agent.messaging.processing import finalize_message, prepare_prompts
from llmling_agent.talk.talk import Talk, TeamTalk
from llmling_agent.utils.now import get_now


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime

    from llmling_agent import MessageNode
    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import PromptCompatible, SupportsStructuredOutput
    from llmling_agent.delegation import AgentPool


logger = get_logger(__name__)

ResultMode = Literal["last", "concat"]


@dataclass(frozen=True, kw_only=True)
class ExtendedTeamTalk(TeamTalk):
    """TeamTalk that also provides TeamRunStats interface."""

    errors: list[tuple[str, str, datetime]] = field(default_factory=list)

    def clear(self) -> None:
        """Reset all tracking data."""
        super().clear()  # Clear base TeamTalk
        self.errors.clear()

    def add_error(self, agent: str, error: str) -> None:
        """Track errors from AgentResponses."""
        self.errors.append((agent, error, get_now()))


class TeamRun[TDeps, TResult](BaseTeam[TDeps, TResult]):
    """Handles team operations with monitoring."""

    @overload  # validator set: it defines the output
    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        shared_prompt: str | None = None,
        validator: MessageNode[Any, TResult],
        picker: SupportsStructuredOutput | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
        agent_pool: AgentPool | None = None,
    ) -> None: ...

    @overload
    def __init__(  # no validator, but all nodes same output type.
        self,
        agents: Sequence[MessageNode[TDeps, TResult]],
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        shared_prompt: str | None = None,
        validator: None = None,
        picker: SupportsStructuredOutput | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
        agent_pool: AgentPool | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        shared_prompt: str | None = None,
        validator: MessageNode[Any, TResult] | None = None,
        picker: SupportsStructuredOutput | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
        agent_pool: AgentPool | None = None,
    ) -> None: ...

    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        shared_prompt: str | None = None,
        validator: MessageNode[Any, TResult] | None = None,
        picker: SupportsStructuredOutput | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
        agent_pool: AgentPool | None = None,
        # result_mode: ResultMode = "last",
    ) -> None:
        super().__init__(
            agents,
            name=name,
            description=description,
            display_name=display_name,
            shared_prompt=shared_prompt,
            picker=picker,
            num_picks=num_picks,
            pick_prompt=pick_prompt,
            agent_pool=agent_pool,
        )
        self.validator = validator
        self.result_mode = "last"

    def __prompt__(self) -> str:
        """Format team info for prompts."""
        members = " -> ".join(a.name for a in self.nodes)
        desc = f" - {self.description}" if self.description else ""
        return f"Sequential Team {self.name!r}{desc}\nPipeline: {members}"

    async def run(
        self,
        *prompts: PromptCompatible | None,
        wait_for_connections: bool | None = None,
        store_history: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Run agents sequentially and return combined message."""
        # Prepare prompts and create user message
        user_msg, processed_prompts, original_message = await prepare_prompts(*prompts)
        self.message_received.emit(user_msg)
        # Execute sequential logic
        message_id = str(uuid4())  # Always generate unique response ID
        result = await self.execute(*processed_prompts, **kwargs)
        all_messages = [r.message for r in result if r.message]
        assert all_messages, "Error during execution, returned None for TeamRun"
        # Determine content based on mode
        match self.result_mode:
            case "last":
                content = all_messages[-1].content
            # case "concat":
            #     content = "\n".join(msg.format() for msg in all_messages)
            case _:
                msg = f"Invalid result mode: {self.result_mode}"
                raise ValueError(msg)

        message = ChatMessage(
            content=content,
            messages=[m for chat_message in all_messages for m in chat_message.messages],
            role="assistant",
            name=self.name,
            associated_messages=all_messages,
            message_id=message_id,
            conversation_id=user_msg.conversation_id,
            metadata={
                "execution_order": [r.agent_name for r in result],
                "start_time": result.start_time.isoformat(),
                "errors": {name: str(error) for name, error in result.errors.items()},
            },
        )

        if store_history:
            pass  # Teams could implement their own history management here if needed
        return await finalize_message(  # Finalize and route message
            message,
            user_msg,
            self,
            self.connections,
            original_message,
            wait_for_connections,
        )

    async def execute(
        self,
        *prompts: PromptCompatible | None,
        **kwargs: Any,
    ) -> TeamResponse[TResult]:
        """Start execution with optional monitoring."""
        self._team_talk.clear()
        start_time = get_now()
        prompts_ = list(prompts)
        if self.shared_prompt:
            prompts_.insert(0, self.shared_prompt)
        responses = [i async for i in self.execute_iter(*prompts_) if isinstance(i, AgentResponse)]
        return TeamResponse(responses, start_time)

    async def run_iter(
        self,
        *prompts: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages from the execution chain."""
        async for item in self.execute_iter(*prompts, **kwargs):
            match item:
                case AgentResponse():
                    if item.message:
                        yield item.message
                case Talk():
                    pass

    async def execute_iter(
        self,
        *prompt: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncIterator[Talk[Any] | AgentResponse[Any]]:
        from toprompt import to_prompt

        connections: list[Talk[Any]] = []
        try:
            combined_prompt = "\n".join([await to_prompt(p) for p in prompt])
            all_nodes = list(await self.pick_agents(combined_prompt))
            if self.validator:
                all_nodes.append(self.validator)
            first = all_nodes[0]
            connections = [s.connect_to(t, queued=True) for s, t in pairwise(all_nodes)]
            for conn in connections:
                self._team_talk.append(conn)

            # First agent
            start = perf_counter()
            message = await first.run(*prompt, **kwargs)
            timing = perf_counter() - start
            response = AgentResponse[Any](first.name, message=message, timing=timing)
            yield response

            # Process through chain
            for connection in connections:
                target = connection.targets[0]
                target_name = target.name
                yield connection

                # Let errors propagate - they break the chain
                start = perf_counter()
                messages = await connection.trigger()

                if target == all_nodes[-1]:
                    last_talk = Talk[Any](target, [], connection_type="run")
                    if response.message:
                        last_talk.stats.messages.append(response.message)
                    self._team_talk.append(last_talk)

                timing = perf_counter() - start
                msg = messages[0]
                response = AgentResponse[Any](target_name, message=msg, timing=timing)
                yield response

        finally:  # Always clean up connections
            for connection in connections:
                connection.disconnect()

    async def run_stream(
        self,
        *prompts: PromptCompatible,
        require_all: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[MessageNode[Any, Any], RichAgentStreamEvent[Any]]]:
        """Stream responses through the chain of team members.

        Args:
            prompts: Input prompts to process through the chain
            require_all: If True, fail if any agent fails. If False,
                         continue with remaining agents.
            kwargs: Additional arguments passed to each agent

        Yields:
            Tuples of (agent, event) where agent is the Agent instance
            and event is the streaming event.
        """
        from llmling_agent.agents.events import StreamCompleteEvent

        current_message = prompts
        collected_content = []
        for agent in self.nodes:
            try:
                agent_content = []

                # Use wrapper to normalize all streaming nodes to (agent, event) tuples
                if not isinstance(agent, SupportsRunStream):
                    msg = f"Agent {agent.name} does not support streaming"
                    raise TypeError(msg)  # noqa: TRY301

                stream = normalize_stream_for_teams(agent, *current_message, **kwargs)

                async for agent_event_tuple in stream:
                    actual_agent, event = agent_event_tuple
                    match event:
                        case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                            agent_content.append(delta)
                            collected_content.append(delta)
                            yield (actual_agent, event)  # Yield tuple with agent context
                        case StreamCompleteEvent(message=message):
                            # Use complete response as input for next agent
                            current_message = (message.content,)
                            yield (actual_agent, event)  # Yield tuple with agent context
                        case _:
                            yield (actual_agent, event)  # Yield tuple with agent context

            except Exception as e:
                if require_all:
                    msg = f"Chain broken at {agent.name}: {e}"
                    logger.exception(msg)
                    raise ValueError(msg) from e
                logger.warning("Chain handler failed", name=agent.name, error=e)


if __name__ == "__main__":

    async def main() -> None:
        from llmling_agent import Agent, Team

        agent1 = Agent(name="Agent1", model="test")
        agent2 = Agent(name="Agent2", model="test")
        agent3 = Agent(name="Agent3", model="test")
        inner_team = Team([agent1, agent2], name="Parallel")
        outer_run = TeamRun([inner_team, agent3], name="Sequential")
        print("Testing TeamRun containing Team...")
        try:
            async for node, event in outer_run.run_stream("test"):
                print(f"{node.name}: {type(event).__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")

    asyncio.run(main())
