"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from asyncio import Lock
from contextlib import AsyncExitStack, asynccontextmanager, suppress
import os
from typing import TYPE_CHECKING, Any, Self, Unpack, overload

from anyenv import ProcessManager
from upath import UPath

from llmling_agent.agent import Agent
from llmling_agent.common_types import NodeName
from llmling_agent.delegation.message_flow_tracker import MessageFlowTracker
from llmling_agent.delegation.team import Team
from llmling_agent.delegation.teamrun import TeamRun
from llmling_agent.log import get_logger
from llmling_agent.messaging import MessageNode
from llmling_agent.talk import TeamTalk
from llmling_agent.talk.registry import ConnectionRegistry
from llmling_agent.tasks import TaskRegistry
from llmling_agent.utils.baseregistry import BaseRegistry
from llmling_agent_config.forward_targets import (
    CallableConnectionConfig,
    FileConnectionConfig,
    NodeConnectionConfig,
)
from llmling_agent_config.workers import AgentWorkerConfig, TeamWorkerConfig


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from psygnal.containers._evented_dict import DictEvents
    from pydantic_ai.output import OutputSpec
    from tokonomics import ModelName
    from upath.types import JoinablePathLike

    from llmling_agent.agent.agent import AgentKwargs
    from llmling_agent.common_types import AgentName, IndividualEventHandler, SessionIdType
    from llmling_agent.delegation.base_team import BaseTeam
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.task import Job


logger = get_logger(__name__)


class AgentPool[TPoolDeps = None](BaseRegistry[NodeName, MessageNode[Any, Any]]):
    """Pool managing message processing nodes (agents and teams).

    Acts as a unified registry for all nodes, providing:
    - Centralized node management and lookup
    - Shared dependency injection
    - Connection management
    - Resource coordination

    Nodes can be accessed through:
    - nodes: All registered nodes (agents and teams)
    - agents: Only Agent instances
    - teams: Only Team instances
    """

    def __init__(
        self,
        manifest: JoinablePathLike | AgentsManifest | None = None,
        *,
        shared_deps_type: type[TPoolDeps] | None = None,
        connect_nodes: bool = True,
        input_provider: InputProvider | None = None,
        parallel_load: bool = True,
        event_handlers: list[IndividualEventHandler] | None = None,
    ):
        """Initialize agent pool with immediate agent creation.

        Args:
            manifest: Agent configuration manifest
            shared_deps_type: Dependencies to share across all nodes
            connect_nodes: Whether to set up forwarding connections
            input_provider: Input provider for tool / step confirmations / HumanAgents
            parallel_load: Whether to load nodes in parallel (async)
            event_handlers: Event handlers to pass through to all agents

        Raises:
            ValueError: If manifest contains invalid node configurations
            RuntimeError: If node initialization fails
        """
        super().__init__()
        from llmling_agent.mcp_server.manager import MCPManager
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent.observability import registry
        from llmling_agent.skills.manager import SkillsManager
        from llmling_agent.storage import StorageManager

        match manifest:
            case None:
                self.manifest = AgentsManifest()
            case str() | os.PathLike() | UPath():
                self.manifest = AgentsManifest.from_file(manifest)
            case AgentsManifest():
                self.manifest = manifest
            case _:
                msg = f"Invalid config path: {manifest}"
                raise ValueError(msg)

        registry.configure_observability(self.manifest.observability)
        self.shared_deps_type = shared_deps_type
        self._input_provider = input_provider
        self.exit_stack = AsyncExitStack()
        self.parallel_load = parallel_load
        self.storage = StorageManager(self.manifest.storage)
        self.event_handlers = event_handlers or []
        self.connection_registry = ConnectionRegistry()
        servers = self.manifest.get_mcp_servers()
        self.mcp = MCPManager(name="pool_mcp", servers=servers, owner="pool")
        self.skills = SkillsManager(name="pool_skills", owner="pool")
        self._tasks = TaskRegistry()
        # Register tasks from manifest
        for name, task in self.manifest.jobs.items():
            self._tasks.register(name, task)
        self.process_manager = ProcessManager()
        self.pool_talk = TeamTalk[Any].from_nodes(list(self.nodes.values()))
        # Create requested agents immediately
        for name in self.manifest.agents:
            agent = self.manifest.get_agent(
                name,
                deps_type=shared_deps_type,
                input_provider=self._input_provider,
                pool=self,
                event_handlers=self.event_handlers,
            )
            self.register(name, agent)

        for agent in self.agents.values():
            self.setup_agent_workers(agent)
        self._create_teams()
        if connect_nodes:
            self._connect_nodes()

        self._enter_lock = Lock()  # Initialize async safety fields
        self._running_count = 0

    async def __aenter__(self) -> Self:
        """Enter async context and initialize all agents."""
        async with self._enter_lock:
            if self._running_count == 0:
                try:
                    # Initialize MCP manager first, then add aggregating provider
                    await self.exit_stack.enter_async_context(self.mcp)
                    await self.exit_stack.enter_async_context(self.skills)
                    aggregating_provider = self.mcp.get_aggregating_provider()
                    skills_provider = self.skills.get_skills_provider()

                    agents = list(self.agents.values())
                    teams = list(self.teams.values())
                    for agent in agents:
                        agent.tools.add_provider(aggregating_provider)
                        agent.tools.add_provider(skills_provider)

                    # Collect remaining components to initialize (MCP already initialized)
                    components: list[AbstractAsyncContextManager[Any]] = [
                        self.storage,
                        *agents,
                        *teams,
                    ]

                    # MCP server is now managed externally - removed from pool
                    # Initialize all components
                    if self.parallel_load:
                        await asyncio.gather(
                            *(self.exit_stack.enter_async_context(c) for c in components)
                        )
                    else:
                        for component in components:
                            await self.exit_stack.enter_async_context(component)

                except Exception as e:
                    await self.cleanup()
                    msg = "Failed to initialize agent pool"
                    logger.exception(msg, exc_info=e)
                    raise RuntimeError(msg) from e
            self._running_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        if self._running_count == 0:
            msg = "AgentPool.__aexit__ called more times than __aenter__"
            raise ValueError(msg)
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0:
                # Remove MCP aggregating provider from all agents
                aggregating_provider = self.mcp.get_aggregating_provider()
                skills_provider = self.skills.get_skills_provider()
                for agent in self.agents.values():
                    agent.tools.remove_provider(aggregating_provider.name)
                    agent.tools.remove_provider(skills_provider.name)
                await self.cleanup()

    @property
    def is_running(self) -> bool:
        """Check if the agent pool is running."""
        return bool(self._running_count)

    async def cleanup(self) -> None:
        """Clean up all agents."""
        # Clean up background processes first
        await self.process_manager.cleanup()
        await self.exit_stack.aclose()
        self.clear()

    @overload
    def create_team_run[TResult](
        self,
        agents: Sequence[str],
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> TeamRun[TPoolDeps, TResult]: ...

    @overload
    def create_team_run[TDeps, TResult](
        self,
        agents: Sequence[MessageNode[TDeps, Any]],
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> TeamRun[TDeps, TResult]: ...

    @overload
    def create_team_run[TResult](
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]],
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> TeamRun[Any, TResult]: ...

    def create_team_run[TResult](
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]] | None = None,
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> TeamRun[Any, TResult]:
        """Create a a sequential TeamRun from a list of Agents.

        Args:
            agents: List of agent names or team/agent instances (all if None)
            validator: Node to validate the results of the TeamRun
            name: Optional name for the team
            description: Optional description for the team
            shared_prompt: Optional prompt for all agents
            picker: Agent to use for picking agents
            num_picks: Number of agents to pick
            pick_prompt: Prompt to use for picking agents
        """
        from llmling_agent.delegation.teamrun import TeamRun

        if agents is None:
            agents = list(self.agents.keys())

        # First resolve/configure agents
        resolved_agents: list[MessageNode[Any, Any]] = []
        for agent in agents:
            if isinstance(agent, str):
                agent = self.get_agent(agent)
            resolved_agents.append(agent)
        team = TeamRun(
            resolved_agents,
            name=name,
            description=description,
            validator=validator,
            shared_prompt=shared_prompt,
            picker=picker,
            num_picks=num_picks,
            pick_prompt=pick_prompt,
        )
        if name:
            self[name] = team
        return team

    @overload
    def create_team(self, agents: Sequence[str]) -> Team[TPoolDeps]: ...

    @overload
    def create_team[TDeps](
        self,
        agents: Sequence[MessageNode[TDeps, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> Team[TDeps]: ...

    @overload
    def create_team(
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> Team[Any]: ...

    def create_team(
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: Agent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> Team[Any]:
        """Create a group from agent names or instances.

        Args:
            agents: List of agent names or instances (all if None)
            name: Optional name for the team
            description: Optional description for the team
            shared_prompt: Optional prompt for all agents
            picker: Agent to use for picking agents
            num_picks: Number of agents to pick
            pick_prompt: Prompt to use for picking agents
        """
        from llmling_agent.delegation.team import Team

        if agents is None:
            agents = list(self.agents.keys())

        resolved_agents = [self.get_agent(i) if isinstance(i, str) else i for i in agents]
        team = Team(
            name=name,
            description=description,
            agents=resolved_agents,
            shared_prompt=shared_prompt,
            picker=picker,
            num_picks=num_picks,
            pick_prompt=pick_prompt,
        )
        if name:
            self[name] = team
        return team

    @asynccontextmanager
    async def track_message_flow(self) -> AsyncIterator[MessageFlowTracker]:
        """Track message flow during a context."""
        tracker = MessageFlowTracker()
        self.connection_registry.message_flow.connect(tracker.track)
        try:
            yield tracker
        finally:
            self.connection_registry.message_flow.disconnect(tracker.track)

    async def run_event_loop(self) -> None:
        """Run pool in event-watching mode until interrupted."""
        print("Starting event watch mode...")
        print("Active nodes: ", ", ".join(self.list_nodes()))
        print("Press Ctrl+C to stop")

        with suppress(KeyboardInterrupt):
            while True:
                await asyncio.sleep(1)

    @property
    def agents(self) -> dict[str, Agent[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        return {i.name: i for i in self._items.values() if isinstance(i, Agent)}

    @property
    def teams(self) -> dict[str, BaseTeam[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        from llmling_agent.delegation.base_team import BaseTeam

        return {i.name: i for i in self._items.values() if isinstance(i, BaseTeam)}

    @property
    def nodes(self) -> dict[str, MessageNode[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        from llmling_agent import MessageNode

        return {i.name: i for i in self._items.values() if isinstance(i, MessageNode)}

    @property
    def node_events(self) -> DictEvents:
        """Get node events."""
        return self._items.events

    def _validate_item(self, item: MessageNode[Any, Any] | Any) -> MessageNode[Any, Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated Node

        Raises:
            LLMlingError: If item is not a valid node
        """
        if not isinstance(item, MessageNode):
            msg = f"Item must be Agent or Team, got {type(item)}"
            raise self._error_class(msg)
        item.context.pool = self
        return item

    def _create_teams(self) -> None:
        """Create all teams in two phases to allow nesting."""
        # Phase 1: Create empty teams

        empty_teams: dict[str, BaseTeam[Any, Any]] = {}
        for name, config in self.manifest.teams.items():
            if config.mode == "parallel":
                empty_teams[name] = Team(
                    [],
                    name=name,
                    display_name=config.display_name,
                    shared_prompt=config.shared_prompt,
                )
            else:
                empty_teams[name] = TeamRun(
                    [],
                    name=name,
                    display_name=config.display_name,
                    shared_prompt=config.shared_prompt,
                )

        # Phase 2: Resolve members
        for name, config in self.manifest.teams.items():
            team = empty_teams[name]
            members: list[MessageNode[Any, Any]] = []
            for member in config.members:
                if member in self.agents:
                    members.append(self.agents[member])
                elif member in empty_teams:
                    members.append(empty_teams[member])
                else:
                    msg = f"Unknown team member: {member}"
                    raise ValueError(msg)
            team.agents.extend(members)
            self[name] = team

    def _connect_nodes(self) -> None:
        """Set up connections defined in manifest."""
        # Merge agent and team configs into one dict of nodes with connections
        for name, config in self.manifest.nodes.items():
            source = self[name]
            for target in config.connections or []:
                match target:
                    case NodeConnectionConfig(name=name_):
                        if name_ not in self:
                            msg = f"Forward target {name_} not found for {name}"
                            raise ValueError(msg)
                        target_node = self[name_]
                    case FileConnectionConfig(path=path_obj):
                        name = f"file_writer_{UPath(path_obj).stem}"
                        target_node = Agent(model=target.get_model(), name=name)
                    case CallableConnectionConfig(callable=fn):
                        target_node = Agent(model=target.get_model(), name=fn.__name__)
                    case _:
                        msg = f"Invalid connection config: {target}"
                        raise ValueError(msg)

                source.connect_to(
                    target_node,
                    connection_type=target.connection_type,
                    name=name,
                    priority=target.priority,
                    delay=target.delay,
                    queued=target.queued,
                    queue_strategy=target.queue_strategy,
                    transform=target.transform,
                    filter_condition=target.filter_condition.check
                    if target.filter_condition
                    else None,
                    stop_condition=target.stop_condition.check if target.stop_condition else None,
                    exit_condition=target.exit_condition.check if target.exit_condition else None,
                )
                source.connections.set_wait_state(
                    target_node,
                    wait=target.wait_for_completion,
                )

    def setup_agent_workers(self, agent: Agent[Any, Any]) -> None:
        """Set up workers for an agent from configuration."""
        for worker_config in self.manifest.agents[agent.name].workers:
            try:
                worker = self.nodes[worker_config.name]
                match worker_config:
                    case TeamWorkerConfig():
                        agent.register_worker(worker)
                    case AgentWorkerConfig():
                        agent.register_worker(
                            worker,
                            reset_history_on_run=worker_config.reset_history_on_run,
                            pass_message_history=worker_config.pass_message_history,
                        )
            except KeyError as e:
                msg = f"Worker agent {worker_config.name!r} not found"
                raise ValueError(msg) from e

    @overload
    def get_agent[TResult = str](
        self,
        agent: AgentName | Agent[Any, str],
        *,
        return_type: type[TResult] = str,  # type: ignore
        model_override: ModelName | str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[TPoolDeps, TResult]: ...

    @overload
    def get_agent[TCustomDeps, TResult = str](
        self,
        agent: AgentName | Agent[Any, str],
        *,
        deps_type: type[TCustomDeps],
        return_type: type[TResult] = str,  # type: ignore
        model_override: ModelName | str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[TCustomDeps, TResult]: ...

    def get_agent(
        self,
        agent: AgentName | Agent[Any, str],
        *,
        deps_type: Any | None = None,
        return_type: Any = str,
        model_override: ModelName | str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[Any, Any]:
        """Get or configure an agent from the pool.

        This method provides flexible agent configuration with dependency injection:
        - Without deps: Agent uses pool's shared dependencies
        - With deps: Agent uses provided custom dependencies

        Args:
            agent: Either agent name or instance
            deps_type: Optional custom dependencies type (overrides shared deps)
            return_type: Optional type for structured responses
            model_override: Optional model override
            session: Optional session ID or query to recover conversation

        Returns:
            Either:
            - Agent[TPoolDeps] when using pool's shared deps
            - Agent[TCustomDeps] when custom deps provided

        Raises:
            KeyError: If agent name not found
            ValueError: If configuration is invalid
        """
        from llmling_agent.agent import Agent

        base = agent if isinstance(agent, Agent) else self.agents[agent]
        # Use custom deps if provided, otherwise use shared deps
        # base.context.data = deps if deps is not None else self.shared_deps
        base.deps_type = deps_type
        base.context.pool = self
        if model_override:
            base.set_model(model_override)
        if session:
            base.conversation.load_history_from_database(session=session)
        if return_type not in {str, None}:
            base.to_structured(return_type)

        return base

    def list_nodes(self) -> list[str]:
        """List available agent names."""
        return list(self.list_items())

    def get_job(self, name: str) -> Job[Any, Any]:
        return self._tasks[name]

    def register_task(self, name: str, task: Job[Any, Any]) -> None:
        self._tasks.register(name, task)

    async def add_agent[TResult = str](
        self,
        name: AgentName,
        *,
        output_type: OutputSpec[TResult] = str,  # type: ignore[assignment]
        **kwargs: Unpack[AgentKwargs],
    ) -> Agent[Any, TResult]:
        """Add a new permanent agent to the pool.

        Args:
            name: Name for the new agent
            output_type: Optional type for structured responses:
            **kwargs: Additional agent configuration

        Returns:
            An agent instance
        """
        from llmling_agent.agent import Agent

        if not kwargs.get("event_handlers"):
            kwargs["event_handlers"] = self.event_handlers
        agent: Agent[Any, TResult] = Agent(
            name=name,
            **kwargs,
            output_type=output_type,
            agent_pool=self,
        )
        # Add MCP aggregating provider from manager
        agent.tools.add_provider(self.mcp.get_aggregating_provider())
        agent.tools.add_provider(self.skills.get_skills_provider())
        agent = await self.exit_stack.enter_async_context(agent)
        self.register(name, agent)
        return agent

    def get_mermaid_diagram(self, include_details: bool = True) -> str:
        """Generate mermaid flowchart of all agents and their connections.

        Args:
            include_details: Whether to show connection details (types, queues, etc)
        """
        lines = ["flowchart LR"]

        # Add all agents as nodes
        for name in self.agents:
            lines.append(f"    {name}[{name}]")  # noqa: PERF401

        # Add all connections as edges
        for agent in self.agents.values():
            connections = agent.connections.get_connections()
            for talk in connections:
                source = talk.source.name
                for target in talk.targets:
                    if include_details:
                        details: list[str] = []
                        details.append(talk.connection_type)
                        if talk.queued:
                            details.append(f"queued({talk.queue_strategy})")
                        if fn := talk.filter_condition:
                            details.append(f"filter:{fn.__name__}")
                        if fn := talk.stop_condition:
                            details.append(f"stop:{fn.__name__}")
                        if fn := talk.exit_condition:
                            details.append(f"exit:{fn.__name__}")

                        label = f"|{' '.join(details)}|" if details else ""
                        lines.append(f"    {source}--{label}-->{target.name}")
                    else:
                        lines.append(f"    {source}-->{target.name}")

        return "\n".join(lines)


if __name__ == "__main__":

    async def main() -> None:
        path = "src/llmling_agent/config_resources/agents.yml"
        async with AgentPool(path) as pool:
            agent = pool.get_agent("overseer")
            print(agent)

    asyncio.run(main())
