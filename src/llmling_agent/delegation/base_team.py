"""Base class for teams."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, overload

from psygnal.containers import EventedList

from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent.talk.stats import AggregatedMessageStats
from llmling_agent.tools.base import Tool
from llmling_agent.utils.inspection import has_return_type
from llmling_agent_config.teams import TeamConfig


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    import os

    import PIL.Image
    from psygnal.containers._evented_list import ListEvents
    from toprompt import AnyPromptType

    from llmling_agent import AgentPool, AnyAgent, Team
    from llmling_agent.common_types import ModelType, ToolType
    from llmling_agent.delegation.teamrun import ExtendedTeamTalk, TeamRun
    from llmling_agent.messaging.messages import ChatMessage, TeamResponse
    from llmling_agent.talk.stats import AggregatedTalkStats
    from llmling_agent_config.mcp_server import MCPServerConfig
    from llmling_agent_config.providers import ProcessorCallback
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_providers.base import AgentProvider

logger = get_logger(__name__)


@dataclass(kw_only=True)
class TeamContext[TDeps](NodeContext[TDeps]):
    """Context for team nodes."""

    config: TeamConfig
    """Current team's specific configuration."""

    pool: AgentPool | None = None
    """Pool the team is part of."""

    @classmethod
    def create_default(
        cls,
        name: str,
        mode: Literal["sequential", "parallel"] = "sequential",
        pool: AgentPool | None = None,
    ) -> TeamContext:
        """Create a default agent context with minimal privileges.

        Args:
            name: Name of the agent
            mode: Execution mode (sequential or parallel)
            pool:(optional): Optional pool the agent is part of
        """
        from llmling_agent_config import TeamConfig

        cfg = TeamConfig(name=name, mode=mode, members=[])
        defn = AgentsManifest()
        return cls(node_name=name, config=cfg, pool=pool, definition=defn)


class BaseTeam[TDeps, TResult](MessageNode[TDeps, TResult]):
    """Base class for Team and TeamRun."""

    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, TResult]],
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        mcp_servers: list[str | MCPServerConfig] | None = None,
        picker: AnyAgent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ):
        """Common variables only for typing."""
        from llmling_agent.delegation.teamrun import ExtendedTeamTalk

        self._name = name or " & ".join([i.name for i in agents])
        self.agents = EventedList[MessageNode]()
        self.agents.events.inserted.connect(self._on_node_added)
        self.agents.events.removed.connect(self._on_node_removed)
        self.agents.events.changed.connect(self._on_node_changed)
        super().__init__(
            name=self._name,
            context=self.context,
            mcp_servers=mcp_servers,
            description=description,
        )
        self.agents.extend(list(agents))
        self._team_talk = ExtendedTeamTalk()
        self.shared_prompt = shared_prompt
        self._main_task: asyncio.Task[Any] | None = None
        self._infinite = False
        self.picker = picker
        self.num_picks = num_picks
        self.pick_prompt = pick_prompt

    def to_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """Create a tool from this agent.

        Args:
            name: Optional tool name override
            description: Optional tool description override
        """
        tool_name = name or f"ask_{self.name}"

        async def wrapped_tool(prompt: str) -> TResult:
            result = await self.run(prompt)
            return result.data

        docstring = description or f"Get expert answer from node {self.name}"
        if self.description:
            docstring = f"{docstring}\n\n{self.description}"

        wrapped_tool.__doc__ = docstring
        wrapped_tool.__name__ = tool_name

        return Tool.from_callable(
            wrapped_tool,
            name_override=tool_name,
            description_override=docstring,
        )

    async def pick_agents(self, task: str) -> Sequence[MessageNode[Any, Any]]:
        """Pick agents to run."""
        if self.picker:
            if self.num_picks == 1:
                result = await self.picker.talk.pick(self, task, self.pick_prompt)
                return [result.selection]
            result = await self.picker.talk.pick_multiple(
                self,
                task,
                min_picks=self.num_picks or 1,
                max_picks=self.num_picks,
                prompt=self.pick_prompt,
            )
            return result.selections
        return list(self.agents)

    def _on_node_changed(self, index: int, old: MessageNode, new: MessageNode):
        """Handle node replacement in the agents list."""
        self._on_node_removed(index, old)
        self._on_node_added(index, new)

    def _on_node_added(self, index: int, node: MessageNode[Any, Any]):
        """Handler for adding nodes to the team."""
        from llmling_agent.agent import Agent, StructuredAgent

        if isinstance(node, Agent | StructuredAgent):
            node.tools.add_provider(self.mcp)
        # TODO: Right now connecting here is not desired since emission means db logging
        # ideally db logging would not rely on the "public" agent signal.

        # node.tool_used.connect(self.tool_used)

    def _on_node_removed(self, index: int, node: MessageNode[Any, Any]):
        """Handler for removing nodes from the team."""
        from llmling_agent.agent import Agent, StructuredAgent

        if isinstance(node, Agent | StructuredAgent):
            node.tools.remove_provider(self.mcp)
        # node.tool_used.disconnect(self.tool_used)

    def __repr__(self) -> str:
        """Create readable representation."""
        members = ", ".join(agent.name for agent in self.agents)
        name = f" ({self.name})" if self.name else ""
        return f"{self.__class__.__name__}[{len(self.agents)}]{name}: {members}"

    def __len__(self) -> int:
        """Get number of team members."""
        return len(self.agents)

    def __iter__(self) -> Iterator[MessageNode[TDeps, TResult]]:
        """Iterate over team members."""
        return iter(self.agents)

    def __getitem__(self, index_or_name: int | str) -> MessageNode[TDeps, TResult]:
        """Get team member by index or name."""
        if isinstance(index_or_name, str):
            return next(agent for agent in self.agents if agent.name == index_or_name)
        return self.agents[index_or_name]

    def __or__(
        self,
        other: AnyAgent[Any, Any] | ProcessorCallback[Any] | BaseTeam[Any, Any],
    ) -> TeamRun[Any, Any]:
        """Create a sequential pipeline."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.teamrun import TeamRun

        # Handle conversion of callables first
        if callable(other):
            if has_return_type(other, str):
                other = Agent.from_callback(other)
            else:
                other = StructuredAgent.from_callback(other)
            other.context.pool = self.context.pool

        # If we're already a TeamRun, extend it
        if isinstance(self, TeamRun):
            if self.validator:
                # If we have a validator, create new TeamRun to preserve validation
                return TeamRun([self, other])
            self.agents.append(other)
            return self
        # Otherwise create new TeamRun
        return TeamRun([self, other])

    @overload
    def __and__(self, other: Team[None]) -> Team[None]: ...

    @overload
    def __and__(self, other: Team[TDeps]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: Team[Any]) -> Team[Any]: ...

    @overload
    def __and__(self, other: AnyAgent[TDeps, Any]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: AnyAgent[Any, Any]) -> Team[Any]: ...

    def __and__(
        self, other: Team[Any] | AnyAgent[Any, Any] | ProcessorCallback[Any]
    ) -> Team[Any]:
        """Combine teams, preserving type safety for same types."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.team import Team

        if callable(other):
            if has_return_type(other, str):
                other = Agent.from_callback(other)
            else:
                other = StructuredAgent.from_callback(other)
            other.context.pool = self.context.pool

        match other:
            case Team():
                # Flatten when combining Teams
                return Team([*self.agents, *other.agents])
            case _:
                # Everything else just becomes a member
                return Team([*self.agents, other])

    async def get_stats(self) -> AggregatedMessageStats:
        """Get aggregated stats from all team members."""
        stats = [await agent.get_stats() for agent in self.agents]
        return AggregatedMessageStats(stats=stats)

    @property
    def is_running(self) -> bool:
        """Whether execution is currently running."""
        return bool(self._main_task and not self._main_task.done())

    def is_busy(self) -> bool:
        """Check if team is processing any tasks."""
        return bool(self.task_manager._pending_tasks or self._main_task)

    async def stop(self):
        """Stop background execution if running."""
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            await self._main_task
        self._main_task = None
        await self.task_manager.cleanup_tasks()

    async def wait(self) -> ChatMessage[Any] | None:
        """Wait for background execution to complete and return last message."""
        if not self._main_task:
            msg = "No execution running"
            raise RuntimeError(msg)
        if self._infinite:
            msg = "Cannot wait on infinite execution"
            raise RuntimeError(msg)
        try:
            return await self._main_task
        finally:
            await self.task_manager.cleanup_tasks()
            self._main_task = None

    async def run_in_background(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        max_count: int | None = 1,  # 1 = single execution, None = indefinite
        interval: float = 1.0,
        **kwargs: Any,
    ) -> ExtendedTeamTalk:
        """Start execution in background.

        Args:
            prompts: Prompts to execute
            max_count: Maximum number of executions (None = run indefinitely)
            interval: Seconds between executions
            **kwargs: Additional args for execute()
        """
        if self._main_task:
            msg = "Execution already running"
            raise RuntimeError(msg)
        self._infinite = max_count is None

        async def _continuous() -> ChatMessage[Any] | None:
            count = 0
            last_message = None
            while max_count is None or count < max_count:
                try:
                    result = await self.execute(*prompts, **kwargs)
                    last_message = result[-1].message if result else None
                    count += 1
                    if max_count is None or count < max_count:
                        await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    logger.debug("Background execution cancelled")
                    break
            return last_message

        self._main_task = self.task_manager.create_task(
            _continuous(), name="main_execution"
        )
        return self._team_talk

    @property
    def execution_stats(self) -> AggregatedTalkStats:
        """Get current execution statistics."""
        return self._team_talk.stats

    @property
    def talk(self) -> ExtendedTeamTalk:
        """Get current connection."""
        return self._team_talk

    @property
    def events(self) -> ListEvents:
        """Get events for the team."""
        return self.agents.events

    async def cancel(self):
        """Cancel execution and cleanup."""
        if self._main_task:
            self._main_task.cancel()
        await self.task_manager.cleanup_tasks()

    def get_structure_diagram(self) -> str:
        """Generate mermaid flowchart of node hierarchy."""
        lines = ["flowchart TD"]

        def add_node(node: MessageNode[Any, Any], parent: str | None = None):
            """Recursively add node and its members to diagram."""
            node_id = f"node_{id(node)}"
            lines.append(f"    {node_id}[{node.name}]")
            if parent:
                lines.append(f"    {parent} --> {node_id}")

            # If it's a team, recursively add its members
            from llmling_agent.delegation.base_team import BaseTeam

            if isinstance(node, BaseTeam):
                for member in node.agents:
                    add_node(member, node_id)

        # Start with root nodes (team members)
        for node in self.agents:
            add_node(node)

        return "\n".join(lines)

    def iter_agents(self) -> Iterator[AnyAgent[Any, Any]]:
        """Recursively iterate over all child agents."""
        from llmling_agent.agent import Agent, StructuredAgent

        for node in self.agents:
            match node:
                case BaseTeam():
                    yield from node.iter_agents()
                case Agent() | StructuredAgent():
                    yield node
                case _:
                    msg = f"Invalid node type: {type(node)}"
                    raise ValueError(msg)

    @property
    def context(self) -> TeamContext:
        """Get shared pool from team members.

        Raises:
            ValueError: If team members belong to different pools
        """
        from llmling_agent.delegation.team import Team

        pool_ids: set[int] = set()
        shared_pool: AgentPool | None = None
        team_config: TeamConfig | None = None

        for agent in self.iter_agents():
            if agent.context and agent.context.pool:
                pool_id = id(agent.context.pool)
                if pool_id not in pool_ids:
                    pool_ids.add(pool_id)
                    shared_pool = agent.context.pool
                    if shared_pool.manifest.teams:
                        team_config = shared_pool.manifest.teams.get(self.name)
        if not team_config:
            mode = "parallel" if isinstance(self, Team) else "sequential"
            team_config = TeamConfig(name=self.name, mode=mode, members=[])
        if not pool_ids:
            logger.info("No pool found for team %s.", self.name)
            return TeamContext(
                node_name=self.name,
                pool=shared_pool,
                config=team_config,
                definition=shared_pool.manifest if shared_pool else AgentsManifest(),
            )

        if len(pool_ids) > 1:
            msg = f"Team members in {self.name} belong to different pools"
            raise ValueError(msg)
        return TeamContext(
            node_name=self.name,
            pool=shared_pool,
            config=team_config,
            definition=shared_pool.manifest if shared_pool else AgentsManifest(),
        )

    @context.setter
    def context(self, value: NodeContext):
        msg = "Cannot set context on BaseTeam"
        raise RuntimeError(msg)

    async def distribute(
        self,
        content: str,
        *,
        tools: list[str] | None = None,
        resources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Distribute content and capabilities to all team members."""
        for agent in self.iter_agents():
            # Add context message
            agent.conversation.add_context_message(
                content, source="distribution", metadata=metadata
            )

            # Register tools if provided
            if tools:
                for tool in tools:
                    agent.tools.register_tool(tool)

            # Load resources if provided
            if resources:
                for resource in resources:
                    await agent.conversation.load_context_source(resource)

    @asynccontextmanager
    async def temporary_state(
        self,
        *,
        system_prompts: list[AnyPromptType] | None = None,
        replace_prompts: bool = False,
        tools: list[ToolType] | None = None,
        replace_tools: bool = False,
        history: list[AnyPromptType] | SessionQuery | None = None,
        replace_history: bool = False,
        pause_routing: bool = False,
        model: ModelType | None = None,
        provider: AgentProvider | None = None,
    ) -> AsyncIterator[Self]:
        """Temporarily modify state of all agents in the team.

        All agents in the team will enter their temporary state simultaneously.

        Args:
            system_prompts: Temporary system prompts to use
            replace_prompts: Whether to replace existing prompts
            tools: Temporary tools to make available
            replace_tools: Whether to replace existing tools
            history: Conversation history (prompts or query)
            replace_history: Whether to replace existing history
            pause_routing: Whether to pause message routing
            model: Temporary model override
            provider: Temporary provider override
        """
        # Get all agents (flattened) before entering context
        agents = list(self.iter_agents())

        async with AsyncExitStack() as stack:
            if pause_routing:
                await stack.enter_async_context(self.connections.paused_routing())
            # Enter temporary state for all agents
            for agent in agents:
                await stack.enter_async_context(
                    agent.temporary_state(
                        system_prompts=system_prompts,
                        replace_prompts=replace_prompts,
                        tools=tools,
                        replace_tools=replace_tools,
                        history=history,
                        replace_history=replace_history,
                        pause_routing=pause_routing,
                        model=model,
                        provider=provider,
                    )
                )
            try:
                yield self
            finally:
                # AsyncExitStack will handle cleanup of all states
                pass

    @abstractmethod
    async def execute(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> TeamResponse: ...

    def run_sync(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            store_history: Whether the message exchange should be added to the
                           context window
        Returns:
            Result containing response and run information
        """
        coro = self.run(*prompt, store_history=store_history)
        return self.task_manager.run_task_sync(coro)


if __name__ == "__main__":

    async def main():
        from llmling_agent import Agent, Team

        agent = Agent("My Agent")
        agent_2 = Agent("My Agent")
        team = Team([agent, agent_2], mcp_servers=["uvx mcp-server-git"])
        async with team:
            print(await agent.tools.get_tools())

    import asyncio

    asyncio.run(main())
