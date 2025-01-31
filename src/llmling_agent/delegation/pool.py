"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, suppress
import signal
from typing import TYPE_CHECKING, Any, Self, Unpack, cast, overload

from llmling import BaseRegistry, LLMLingError
from typing_extensions import TypeVar

from llmling_agent.agent import AnyAgent
from llmling_agent.common_types import AgentName
from llmling_agent.delegation.team import Team
from llmling_agent.delegation.teamrun import TeamRun
from llmling_agent.log import get_logger
from llmling_agent.mcp_server.manager import MCPManager
from llmling_agent.models.forward_targets import (
    CallableConnectionConfig,
    FileConnectionConfig,
    NodeConnectionConfig,
)
from llmling_agent.talk import Talk, TeamTalk
from llmling_agent.talk.talk import ConnectionRegistry
from llmling_agent.tasks import TaskRegistry


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from psygnal.containers import EventedDict

    from llmling_agent.agent import Agent, StructuredAgent
    from llmling_agent.agent.agent import AgentKwargs
    from llmling_agent.common_types import SessionIdType, StrPath
    from llmling_agent.delegation.base_team import BaseTeam
    from llmling_agent.messaging.messagenode import MessageNode
    from llmling_agent.models.agents import AgentsManifest, WorkerConfig
    from llmling_agent.models.context import ConfirmationCallback
    from llmling_agent.models.session import SessionQuery
    from llmling_agent.models.task import Job
    from llmling_agent.responses.models import ResponseDefinition


logger = get_logger(__name__)


TResult = TypeVar("TResult", default=Any)
TPoolDeps = TypeVar("TPoolDeps", default=None)


class AgentPool[TPoolDeps](BaseRegistry[AgentName, AnyAgent[Any, Any]]):
    """Pool for managing multiple agents with shared dependencies.

    The pool acts as a central registry and dependency provider for agents.
    By default, all agents share the pool's dependencies, but individual
    agents can override with custom dependencies.

    Generic Parameters:
        TPoolDeps: Type of shared dependencies used across agents.
                   Can be None if no shared dependencies are needed.
    """

    def __init__(
        self,
        manifest: StrPath | AgentsManifest | None = None,
        *,
        shared_deps: TPoolDeps | None = None,
        nodes_to_load: list[AgentName] | None = None,
        connect_nodes: bool = True,
        confirmation_callback: ConfirmationCallback | None = None,
        parallel_agent_load: bool = True,
    ):
        """Initialize agent pool with immediate agent creation.

        Args:
            manifest: Agent configuration manifest
            shared_deps: Dependencies to share across all agents
            nodes_to_load: Optional list of agent names to initialize
                          If None, all agents from manifest are loaded
            connect_nodes: Whether to set up forwarding connections
            confirmation_callback: Handler callback for tool / step confirmations
            parallel_agent_load: Whether to load agents in parallel (async)

        Raises:
            ValueError: If manifest contains invalid agent configurations
            RuntimeError: If agent initialization fails
        """
        super().__init__()
        from llmling_agent.models.agents import AgentsManifest
        from llmling_agent.storage import StorageManager

        match manifest:
            case None:
                self.manifest = AgentsManifest()
            case str():
                self.manifest = AgentsManifest.from_file(manifest)
            case AgentsManifest():
                self.manifest = manifest
            case _:
                msg = f"Invalid config path: {manifest}"
                raise ValueError(msg)
        self.shared_deps = shared_deps
        self._confirmation_callback = confirmation_callback
        self.exit_stack = AsyncExitStack()
        self.parallel_agent_load = parallel_agent_load
        self.storage = StorageManager(self.manifest.storage)
        self._teams: dict[str, BaseTeam[Any, Any]] = {}
        self.connection_registry = ConnectionRegistry()
        self.mcp = MCPManager(self.manifest.get_mcp_servers())
        # Validate requested agents exist
        to_load = set(nodes_to_load) if nodes_to_load else set(self.manifest.agents)
        if invalid := (to_load - set(self.manifest.agents)):
            msg = f"Unknown agents: {', '.join(invalid)}"
            raise ValueError(msg)
        # register tasks
        self._tasks = TaskRegistry()
        # Register tasks from manifest
        for name, task in self.manifest.jobs.items():
            self._tasks.register(name, task)
        self.pool_talk = TeamTalk[Any].from_agents(list(self.agents.values()))

        # Create requested agents immediately
        for name in to_load:
            agent = self.manifest.get_agent(name, deps=shared_deps)
            self.register(name, agent)

        # Then set up worker relationships
        for name, config in self.manifest.agents.items():
            if name in self and config.workers:
                self.setup_agent_workers(self[name], config.workers)

        # Set up forwarding connections
        if connect_nodes:
            self._connect_nodes()

    async def __aenter__(self) -> Self:
        """Enter async context and initialize all agents."""
        try:
            # Add MCP tool provider to all agents
            # Initialize MCP and agents through exit stack
            agents = list(self.agents.values())
            teams = list(self._teams.values())
            for agent in agents:
                agent.tools.add_provider(self.mcp)
            if self.parallel_agent_load:
                await asyncio.gather(
                    self.exit_stack.enter_async_context(self.mcp),
                    *(self.exit_stack.enter_async_context(a) for a in agents),
                    *(self.exit_stack.enter_async_context(t) for t in teams),
                )
            else:
                await self.exit_stack.enter_async_context(self.mcp)
                for agent in agents:
                    await self.exit_stack.enter_async_context(agent)
                for team in teams:
                    await self.exit_stack.enter_async_context(team)
        except Exception as e:
            await self.cleanup()
            msg = "Failed to initialize agent pool"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit async context."""
        # Remove MCP tool provider from all agents
        for agent in self.agents.values():
            agent.tools.remove_provider(self.mcp)
        await self.cleanup()

    async def cleanup(self):
        """Clean up all agents."""
        for agent in self.values():
            if agent.runtime:
                await agent.runtime.shutdown()
        await self.exit_stack.aclose()
        self.clear()

    @overload
    def create_team_run(
        self,
        agents: Sequence[str],
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: AnyAgent[Any, Any] | None = None,
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
        picker: AnyAgent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> TeamRun[TDeps, TResult]: ...

    @overload
    def create_team_run(
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]],
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: AnyAgent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
    ) -> TeamRun[Any, TResult]: ...

    def create_team_run(
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]] | None = None,
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        picker: AnyAgent[Any, Any] | None = None,
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
            self._teams[name] = team
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
        picker: AnyAgent[Any, Any] | None = None,
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
        picker: AnyAgent[Any, Any] | None = None,
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
        picker: AnyAgent[Any, Any] | None = None,
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

        # First resolve/configure agents
        resolved_agents: list[MessageNode[Any, Any]] = []
        for agent in agents:
            if isinstance(agent, str):
                agent = self.get_agent(agent)
            resolved_agents.append(agent)

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
            self._teams[name] = team
        return team

    async def run_event_loop(self) -> None:
        """Run pool in event-watching mode until interrupted."""
        import sys

        print("Starting event watch mode...")
        print("Active agents: ", ", ".join(self.list_agents()))
        print("Press Ctrl+C to stop")

        stop_event = asyncio.Event()

        if sys.platform != "win32":
            # Unix: Use signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop_event.set)
            while True:
                await asyncio.sleep(1)
        else:
            # Windows: Use keyboard interrupt
            with suppress(KeyboardInterrupt):
                while True:
                    await asyncio.sleep(1)

    @property
    def agents(self) -> EventedDict[str, AnyAgent[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        return self._items

    @property
    def _error_class(self) -> type[LLMLingError]:
        """Error class for agent operations."""
        return LLMLingError

    def _validate_item(self, item: AnyAgent[Any, Any] | Any) -> AnyAgent[Any, Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated Agent

        Raises:
            LLMlingError: If item is not a valid agent
        """
        from llmling_agent.agent import Agent, StructuredAgent

        if not isinstance(item, Agent | StructuredAgent):
            msg = f"Item must be Agent, got {type(item)}"
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
                    [], name=name, shared_prompt=config.shared_prompt
                )
            else:
                empty_teams[name] = TeamRun(
                    [], name=name, shared_prompt=config.shared_prompt
                )

        # Phase 2: Resolve members
        for name, config in self.manifest.teams.items():
            team = empty_teams[name]
            members: list[BaseTeam[Any, Any] | AnyAgent[Any, Any]] = []
            for member in config.members:
                if member in self.agents:
                    members.append(self.agents[member])
                elif member in empty_teams:
                    members.append(empty_teams[member])
                else:
                    msg = f"Unknown team member: {member}"
                    raise ValueError(msg)
            team.agents.extend(members)

        self._teams = empty_teams

    def _connect_nodes(self) -> None:
        """Set up connections defined in manifest."""
        # Merge agent and team configs into one dict of nodes with connections
        node_map = {
            name: cfg
            for collection in (self.manifest.agents, self.manifest.teams)
            for name, cfg in collection.items()
            if cfg.connections
        }

        for name, config in node_map.items():
            source = self[name] if name in self else self._teams[name]
            for target in config.connections or []:
                match target:
                    case NodeConnectionConfig():
                        if target.name not in self and target.name not in self._teams:
                            msg = f"Forward target {target.name} not found for {name}"
                            raise ValueError(msg)
                        target_node = (
                            self[target.name]
                            if target.name in self
                            else self._teams[target.name]
                        )
                    case FileConnectionConfig() | CallableConnectionConfig():
                        target_node = Agent(provider=target.get_provider())
                    case _:
                        msg = f"Invalid connection config: {target}"
                        raise ValueError(msg)

                source.connect_to(
                    target_node,  # type: ignore  # recognized as "Any | BaseTeam[Any, Any]" by mypy?
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
                    stop_condition=target.stop_condition.check
                    if target.stop_condition
                    else None,
                    exit_condition=target.exit_condition.check
                    if target.exit_condition
                    else None,
                )
                source.connections.set_wait_state(
                    target_node,
                    wait=target.wait_for_completion,
                )

    @overload
    async def clone_agent[TDeps](
        self,
        agent: AgentName | Agent[TDeps],
        new_name: AgentName | None = None,
        *,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> Agent[TDeps]: ...

    @overload
    async def clone_agent[TDeps, TResult](
        self,
        agent: StructuredAgent[TDeps, TResult],
        new_name: AgentName | None = None,
        *,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> StructuredAgent[TDeps, TResult]: ...

    async def clone_agent[TDeps, TAgentResult](
        self,
        agent: AgentName | AnyAgent[TDeps, TAgentResult],
        new_name: AgentName | None = None,
        *,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> AnyAgent[TDeps, TAgentResult]:
        """Create a copy of an agent.

        Args:
            agent: Agent instance or name to clone
            new_name: Optional name for the clone
            system_prompts: Optional different prompts
            template_context: Variables for template rendering

        Returns:
            The new agent instance
        """
        from llmling_agent.agent import Agent, StructuredAgent

        # Get original config
        if isinstance(agent, str):
            if agent not in self.manifest.agents:
                msg = f"Agent {agent} not found"
                raise KeyError(msg)
            config = self.manifest.agents[agent]
            original_agent: AnyAgent[Any, Any] = self.get_agent(agent)
        else:
            config = agent.context.config  # type: ignore
            original_agent = agent

        # Create new config
        new_config = config.model_copy(deep=True)

        # Apply overrides
        if system_prompts:
            new_config.system_prompts = system_prompts

        # Handle template rendering
        if template_context:
            new_config.system_prompts = new_config.render_system_prompts(template_context)

        # Create new agent with same runtime
        new_agent = Agent[TDeps](
            runtime=original_agent.runtime,
            context=original_agent.context,
            # result_type=original_agent.actual_type,
            model=new_config.model,
            system_prompt=new_config.system_prompts,
            name=new_name or f"{config.name}_copy_{len(self.agents)}",
        )
        if isinstance(original_agent, StructuredAgent):
            new_agent = new_agent.to_structured(original_agent.actual_type)

        # Register in pool
        agent_name = new_agent.name
        self.manifest.agents[agent_name] = new_config
        self.agents[agent_name] = new_agent
        return await self.exit_stack.enter_async_context(new_agent)

    @overload
    async def create_agent(
        self,
        name: AgentName,
        *,
        session: SessionIdType | SessionQuery = None,
        name_override: str | None = None,
    ) -> Agent[TPoolDeps]: ...

    @overload
    async def create_agent[TCustomDeps](
        self,
        name: AgentName,
        *,
        deps: TCustomDeps,
        session: SessionIdType | SessionQuery = None,
        name_override: str | None = None,
    ) -> Agent[TCustomDeps]: ...

    @overload
    async def create_agent[TResult](
        self,
        name: AgentName,
        *,
        return_type: type[TResult],
        session: SessionIdType | SessionQuery = None,
        name_override: str | None = None,
    ) -> StructuredAgent[TPoolDeps, TResult]: ...

    @overload
    async def create_agent[TCustomDeps, TResult](
        self,
        name: AgentName,
        *,
        deps: TCustomDeps,
        return_type: type[TResult],
        session: SessionIdType | SessionQuery = None,
        name_override: str | None = None,
    ) -> StructuredAgent[TCustomDeps, TResult]: ...

    async def create_agent(
        self,
        name: AgentName,
        *,
        deps: Any | None = None,
        return_type: Any | None = None,
        session: SessionIdType | SessionQuery = None,
        name_override: str | None = None,
    ) -> AnyAgent[Any, Any]:
        """Create a new agent instance from configuration.

        Args:
            name: Name of the agent configuration to use
            deps: Optional custom dependencies (overrides pool deps)
            return_type: Optional type for structured responses
            session: Optional session ID or query to recover conversation
            name_override: Optional different name for this instance

        Returns:
            New agent instance with the specified configuration

        Raises:
            KeyError: If agent configuration not found
            ValueError: If configuration is invalid
        """
        if name not in self.manifest.agents:
            msg = f"Agent configuration {name!r} not found"
            raise KeyError(msg)

        # Use Manifest.get_agent for proper initialization
        final_deps = deps if deps is not None else self.shared_deps
        agent = self.manifest.get_agent(name, deps=final_deps)
        # Override name if requested
        if name_override:
            agent.name = name_override

        # Set pool reference
        agent.context.pool = self

        # Handle session if provided
        if session:
            agent.conversation.load_history_from_database(session=session)

        # Initialize agent through exit stack
        agent = await self.exit_stack.enter_async_context(agent)

        # Override structured configuration if provided
        if return_type is not None:
            return agent.to_structured(return_type)

        return agent

    def setup_agent_workers(self, agent: AnyAgent[Any, Any], workers: list[WorkerConfig]):
        """Set up workers for an agent from configuration."""
        for worker_config in workers:
            try:
                worker = self.get_agent(worker_config.name)
                agent.register_worker(
                    worker,
                    name=worker_config.name,
                    reset_history_on_run=worker_config.reset_history_on_run,
                    pass_message_history=worker_config.pass_message_history,
                    share_context=worker_config.share_context,
                )
            except KeyError as e:
                msg = f"Worker agent {worker_config.name!r} not found"
                raise ValueError(msg) from e

    @overload
    def get_agent(
        self,
        agent: AgentName | Agent[Any],
        *,
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[TPoolDeps]: ...

    @overload
    def get_agent[TResult](
        self,
        agent: AgentName | Agent[Any],
        *,
        return_type: type[TResult],
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> StructuredAgent[TPoolDeps, TResult]: ...

    @overload
    def get_agent[TCustomDeps](
        self,
        agent: AgentName | Agent[Any],
        *,
        deps: TCustomDeps,
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[TCustomDeps]: ...

    @overload
    def get_agent[TCustomDeps, TResult](
        self,
        agent: AgentName | Agent[Any],
        *,
        deps: TCustomDeps,
        return_type: type[TResult],
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> StructuredAgent[TCustomDeps, TResult]: ...

    def get_agent(
        self,
        agent: AgentName | Agent[Any],
        *,
        deps: Any | None = None,
        return_type: Any | None = None,
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> AnyAgent[Any, Any]:
        """Get or configure an agent from the pool.

        This method provides flexible agent configuration with dependency injection:
        - Without deps: Agent uses pool's shared dependencies
        - With deps: Agent uses provided custom dependencies
        - With return_type: Returns a StructuredAgent with type validation

        Args:
            agent: Either agent name or instance
            deps: Optional custom dependencies (overrides shared deps)
            return_type: Optional type for structured responses
            model_override: Optional model override
            session: Optional session ID or query to recover conversation

        Returns:
            Either:
            - Agent[TPoolDeps] when using pool's shared deps
            - Agent[TCustomDeps] when custom deps provided
            - StructuredAgent when return_type provided

        Raises:
            KeyError: If agent name not found
            ValueError: If configuration is invalid
        """
        from llmling_agent.agent import Agent
        from llmling_agent.models.context import AgentContext

        # Get base agent
        base = agent if isinstance(agent, Agent) else self.agents[agent]

        # Setup context and dependencies
        if base.context is None:
            base.context = AgentContext[Any].create_default(base.name)

        # Use custom deps if provided, otherwise use shared deps
        base.context.data = deps if deps is not None else self.shared_deps
        base.context.pool = self

        # Apply overrides
        if model_override:
            base.set_model(model_override)

        if session:
            base.conversation.load_history_from_database(session=session)

        # Convert to structured if needed
        if return_type is not None:
            return base.to_structured(return_type)

        return base

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.list_items())

    def get_job(self, name: str) -> Job[Any, Any]:
        return self._tasks[name]

    def register_task(self, name: str, task: Job[Any, Any]):
        self._tasks.register(name, task)

    @overload
    async def add_agent(
        self,
        name: AgentName,
        *,
        result_type: None = None,
        **kwargs: Unpack[AgentKwargs],
    ) -> Agent[Any]: ...

    @overload
    async def add_agent[TResult](
        self,
        name: AgentName,
        *,
        result_type: type[TResult] | str | ResponseDefinition,
        **kwargs: Unpack[AgentKwargs],
    ) -> StructuredAgent[Any, TResult]: ...

    async def add_agent(
        self,
        name: AgentName,
        *,
        result_type: type[Any] | str | ResponseDefinition | None = None,
        **kwargs: Unpack[AgentKwargs],
    ) -> Agent[Any] | StructuredAgent[Any, Any]:
        """Add a new permanent agent to the pool.

        Args:
            name: Name for the new agent
            result_type: Optional type for structured responses:
                - None: Regular unstructured agent
                - type: Python type for validation
                - str: Name of response definition
                - ResponseDefinition: Complete response definition
            **kwargs: Additional agent configuration

        Returns:
            Either a regular Agent or StructuredAgent depending on result_type
        """
        from llmling_agent.agent import Agent

        agent: AnyAgent[Any, Any] = Agent(name=name, **kwargs)
        agent.tools.add_provider(self.mcp)
        agent = await self.exit_stack.enter_async_context(agent)
        # Convert to structured if needed
        if result_type is not None:
            agent = agent.to_structured(result_type)
        self.register(name, agent)
        return agent

    def get_mermaid_diagram(
        self,
        include_details: bool = True,
    ) -> str:
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
                talk = cast(Talk[Any], talk)  # help mypy understand it's a Talk
                source = talk.source.name
                for target in talk.targets:
                    if include_details:
                        details: list[str] = []
                        details.append(talk.connection_type)
                        if talk.queued:
                            details.append(f"queued({talk.queue_strategy})")
                        if fn := talk._filter_condition:  # type: ignore
                            details.append(f"filter:{fn.__name__}")
                        if fn := talk._stop_condition:  # type: ignore
                            details.append(f"stop:{fn.__name__}")
                        if fn := talk._exit_condition:  # type: ignore
                            details.append(f"exit:{fn.__name__}")

                        label = f"|{' '.join(details)}|" if details else ""
                        lines.append(f"    {source}--{label}-->{target.name}")
                    else:
                        lines.append(f"    {source}-->{target.name}")

        return "\n".join(lines)


if __name__ == "__main__":
    from llmling_agent.agent import Agent

    async def main():
        path = "src/llmling_agent/config/resources/agents.yml"
        async with AgentPool[None](path) as pool:
            agent: Agent[Any] = pool.get_agent("overseer")
            print(agent)

    import asyncio

    asyncio.run(main())
