"""Base class for teams."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, overload

import anyio
from psygnal.containers import EventedList

from agentpool.log import get_logger
from agentpool.messaging import MessageNode
from agentpool.messaging.context import NodeContext
from agentpool.talk.stats import AggregatedMessageStats


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from evented_config import EventConfig
    from psygnal.containers._evented_list import ListEvents
    from toprompt import AnyPromptType

    from agentpool import Agent, AgentPool, Team
    from agentpool.common_types import (
        ModelType,
        ProcessorCallback,
        PromptCompatible,
        ToolType,
    )
    from agentpool.delegation.teamrun import ExtendedTeamTalk, TeamRun
    from agentpool.messaging import ChatMessage, TeamResponse
    from agentpool.talk.stats import AggregatedTalkStats
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.session import SessionQuery

logger = get_logger(__name__)


@dataclass(kw_only=True)
class TeamContext[TDeps = object](NodeContext[TDeps]):
    """Context for team nodes."""

    pool: AgentPool | None = None
    """Pool the team is part of."""


class BaseTeam[TDeps, TResult](MessageNode[TDeps, TResult]):
    """Base class for Team and TeamRun."""

    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, TResult]],
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        shared_prompt: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        event_configs: Sequence[EventConfig] | None = None,
        agent_pool: AgentPool | None = None,
    ) -> None:
        """Common variables only for typing."""
        from agentpool.delegation.teamrun import ExtendedTeamTalk

        self._name = name or " & ".join([i.name for i in agents])
        self.nodes = EventedList[MessageNode[Any, Any]]()
        self.nodes.events.inserted.connect(self._on_node_added)
        self.nodes.events.removed.connect(self._on_node_removed)
        self.nodes.events.changed.connect(self._on_node_changed)
        super().__init__(
            name=self._name,
            display_name=display_name,
            mcp_servers=mcp_servers,
            description=description,
            event_configs=event_configs,
            agent_pool=agent_pool,
        )
        self.nodes.extend(list(agents))
        self._team_talk = ExtendedTeamTalk()
        self.shared_prompt = shared_prompt
        self._main_task: asyncio.Task[ChatMessage[Any] | None] | None = None
        self._infinite = False

    def _on_node_changed(
        self, index: int, old: MessageNode[Any, Any], new: MessageNode[Any, Any]
    ) -> None:
        """Handle node replacement in the agents list."""
        self._on_node_removed(index, old)
        self._on_node_added(index, new)

    def _on_node_added(self, index: int, node: MessageNode[Any, Any]) -> None:
        """Handler for adding new nodes to the team."""
        from agentpool.agents import Agent

        if isinstance(node, Agent):
            # Add MCP aggregating provider from manager
            aggregating_provider = self.mcp.get_aggregating_provider()
            node.tools.add_provider(aggregating_provider)

    def _on_node_removed(self, index: int, node: MessageNode[Any, Any]) -> None:
        """Handler for removing nodes from the team."""
        from agentpool.agents import Agent

        if isinstance(node, Agent):
            # Remove MCP aggregating provider
            aggregating_provider = self.mcp.get_aggregating_provider()
            node.tools.remove_provider(aggregating_provider.name)

    def __repr__(self) -> str:
        """Create readable representation."""
        members = ", ".join(node.name for node in self.nodes)
        name = f" ({self.name})" if self.name else ""
        return f"{self.__class__.__name__}[{len(self.nodes)}]{name}: {members}"

    def __len__(self) -> int:
        """Get number of team members."""
        return len(self.nodes)

    def __iter__(self) -> Iterator[MessageNode[TDeps, TResult]]:
        """Iterate over team members."""
        return iter(self.nodes)

    def __getitem__(self, index_or_name: int | str) -> MessageNode[TDeps, TResult]:
        """Get team member by index or name."""
        if isinstance(index_or_name, str):
            return next(node for node in self.nodes if node.name == index_or_name)
        return self.nodes[index_or_name]

    def __or__(
        self,
        other: Agent[Any, Any] | ProcessorCallback[Any] | BaseTeam[Any, Any],
    ) -> TeamRun[Any, Any]:
        """Create a sequential pipeline."""
        from agentpool.agents import Agent
        from agentpool.delegation.teamrun import TeamRun

        # Handle conversion of callables first
        if callable(other):
            other = Agent.from_callback(other, agent_pool=self.agent_pool)

        # If we're already a TeamRun, extend it
        if isinstance(self, TeamRun):
            if self.validator:
                # If we have a validator, create new TeamRun to preserve validation
                return TeamRun([self, other])
            self.nodes.append(other)
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
    def __and__(self, other: Agent[TDeps, Any]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: Agent[Any, Any]) -> Team[Any]: ...

    def __and__(self, other: Team[Any] | Agent[Any, Any] | ProcessorCallback[Any]) -> Team[Any]:
        """Combine teams, preserving type safety for same types."""
        from agentpool.agents import Agent
        from agentpool.delegation.team import Team

        if callable(other):
            other = Agent.from_callback(other, agent_pool=self.agent_pool)

        match other:
            case Team():
                # Flatten when combining Teams
                return Team([*self.nodes, *other.nodes])
            case _:
                # Everything else just becomes a member
                return Team([*self.nodes, other])

    async def get_stats(self) -> AggregatedMessageStats:
        """Get aggregated stats from all team members."""
        stats = [await node.get_stats() for node in self.nodes]
        return AggregatedMessageStats(stats=stats)

    @property
    def is_running(self) -> bool:
        """Whether execution is currently running."""
        return bool(self._main_task and not self._main_task.done())

    def is_busy(self) -> bool:
        """Check if team is processing any tasks."""
        return bool(self.task_manager._pending_tasks or self._main_task)

    async def stop(self) -> None:
        """Stop background execution if running."""
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            await self._main_task
        self._main_task = None
        await self.task_manager.cleanup_tasks()

    async def wait(self) -> ChatMessage[Any] | None:
        """Wait for background execution to complete and return last message."""
        if not self._main_task:
            raise RuntimeError("No execution running")
        if self._infinite:
            raise RuntimeError("Cannot wait on infinite execution")
        try:
            return await self._main_task
        finally:
            await self.task_manager.cleanup_tasks()
            self._main_task = None

    async def run_in_background(
        self,
        *prompts: PromptCompatible | None,
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
            raise RuntimeError("Execution already running")
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
                        await anyio.sleep(interval)
                except asyncio.CancelledError:
                    logger.debug("Background execution cancelled")
                    break
            return last_message

        self._main_task = self.task_manager.create_task(_continuous(), name="main_execution")
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
        return self.nodes.events

    async def cancel(self) -> None:
        """Cancel execution and cleanup."""
        if self._main_task:
            self._main_task.cancel()
        await self.task_manager.cleanup_tasks()

    def get_structure_diagram(self) -> str:
        """Generate mermaid flowchart of node hierarchy."""
        lines = ["flowchart TD"]

        def add_node(node: MessageNode[Any, Any], parent: str | None = None) -> None:
            """Recursively add node and its members to diagram."""
            node_id = f"node_{id(node)}"
            lines.append(f"    {node_id}[{node.name}]")
            if parent:
                lines.append(f"    {parent} --> {node_id}")

            # If it's a team, recursively add its members
            from agentpool.delegation.base_team import BaseTeam

            if isinstance(node, BaseTeam):
                for member in node.nodes:
                    add_node(member, node_id)

        # Start with root nodes (team members)
        for node in self.nodes:
            add_node(node)

        return "\n".join(lines)

    def iter_agents(self) -> Iterator[Agent[Any, Any]]:
        """Recursively iterate over all child agents."""
        from agentpool.agents import Agent

        for node in self.nodes:
            match node:
                case BaseTeam():
                    yield from node.iter_agents()
                case Agent():
                    yield node
                case _:
                    raise ValueError(f"Invalid node type: {type(node)}")

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
    ) -> TeamContext:
        """Create a new context for this team.

        Args:
            data: Optional custom data to attach to the context
            input_provider: Optional input provider override

        Returns:
            A new TeamContext instance

        Raises:
            ValueError: If team members belong to different pools
        """
        pool_ids: set[int] = set()
        shared_pool: AgentPool | None = None

        for agent in self.iter_agents():
            pool = agent.agent_pool
            if pool:
                pool_id = id(pool)
                if pool_id not in pool_ids:
                    pool_ids.add(pool_id)
                    shared_pool = pool

        if not pool_ids:
            logger.debug("No pool found for team.", team=self.name)
            return TeamContext(
                node=self,
                pool=shared_pool,
                input_provider=input_provider,
                data=data,
            )

        if len(pool_ids) > 1:
            raise ValueError(f"Team members in {self.name} belong to different pools")
        return TeamContext(
            node=self,
            pool=shared_pool,
            input_provider=input_provider,
            data=data,
        )

    @asynccontextmanager
    async def temporary_state(
        self,
        *,
        tools: list[ToolType] | None = None,
        replace_tools: bool = False,
        history: list[AnyPromptType] | SessionQuery | None = None,
        replace_history: bool = False,
        pause_routing: bool = False,
        model: ModelType | None = None,
    ) -> AsyncIterator[Self]:
        """Temporarily modify state of all agents in the team.

        All agents in the team will enter their temporary state simultaneously.

        Args:
            tools: Temporary tools to make available
            replace_tools: Whether to replace existing tools
            history: Conversation history (prompts or query)
            replace_history: Whether to replace existing history
            pause_routing: Whether to pause message routing
            model: Temporary model override
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
                        tools=tools,
                        replace_tools=replace_tools,
                        history=history,
                        replace_history=replace_history,
                        pause_routing=pause_routing,
                        model=model,
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
        *prompts: PromptCompatible | None,
        **kwargs: Any,
    ) -> TeamResponse: ...


if __name__ == "__main__":

    async def main() -> None:
        from agentpool import Agent, Team

        agent = Agent("My Agent", model="openai:gpt-5-nano")
        agent_2 = Agent("My Agent", model="openai:gpt-5-nano")
        team = Team([agent, agent_2], mcp_servers=["uvx mcp-server-git"])
        async with team:
            print(await agent.tools.get_tools())

    anyio.run(main)
