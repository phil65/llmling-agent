"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
import os
from typing import TYPE_CHECKING, Any, Self, TypeVar, Unpack, cast, overload

from anyenv import MultiEventHandler
from llmling import BaseRegistry, LLMLingError
from upath import UPath

from llmling_agent.agent import Agent, StructuredAgent
from llmling_agent.common_types import NodeName
from llmling_agent.delegation.message_flow_tracker import MessageFlowTracker
from llmling_agent.delegation.team import Team
from llmling_agent.delegation.teamrun import TeamRun
from llmling_agent.log import get_logger
from llmling_agent.messaging.context import ProgressCallback
from llmling_agent.messaging.messageemitter import MessageEmitter
from llmling_agent.talk import Talk, TeamTalk
from llmling_agent.talk.registry import ConnectionRegistry
from llmling_agent.tasks import TaskRegistry
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
    from upath.types import JoinablePathLike

    from llmling_agent.agent import AnyAgent
    from llmling_agent.agent.agent import AgentKwargs
    from llmling_agent.common_types import AgentName, SessionIdType
    from llmling_agent.delegation.base_team import BaseTeam
    from llmling_agent.messaging.eventnode import EventNode
    from llmling_agent.messaging.messagenode import MessageNode
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent_config.result_types import StructuredResponseConfig
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.task import Job
    from llmling_agent_input.base import InputProvider


logger = get_logger(__name__)


TResult = TypeVar("TResult", default=Any)


class AgentPool[TPoolDeps = None](BaseRegistry[NodeName, MessageEmitter[Any, Any]]):
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
        shared_deps: TPoolDeps | None = None,
        connect_nodes: bool = True,
        input_provider: InputProvider | None = None,
        parallel_load: bool = True,
        progress_handlers: list[ProgressCallback] | None = None,
    ):
        """Initialize agent pool with immediate agent creation.

        Args:
            manifest: Agent configuration manifest
            shared_deps: Dependencies to share across all nodes
            connect_nodes: Whether to set up forwarding connections
            input_provider: Input provider for tool / step confirmations / HumanAgents
            parallel_load: Whether to load nodes in parallel (async)
            progress_handlers: List of progress handlers to notify about progress

        Raises:
            ValueError: If manifest contains invalid node configurations
            RuntimeError: If node initialization fails
        """
        super().__init__()
        from llmling_agent.mcp_server.manager import MCPManager
        from llmling_agent.models.manifest import AgentsManifest
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
        self.shared_deps = shared_deps
        self._input_provider = input_provider
        self.exit_stack = AsyncExitStack()
        self.parallel_load = parallel_load
        self.storage = StorageManager(self.manifest.storage)
        self.progress_handlers = MultiEventHandler[ProgressCallback](progress_handlers)
        self.connection_registry = ConnectionRegistry()
        servers = self.manifest.get_mcp_servers()
        self.mcp = MCPManager(name="pool_mcp", servers=servers, owner="pool")
        self._tasks = TaskRegistry()
        # Register tasks from manifest
        for name, task in self.manifest.jobs.items():
            self._tasks.register(name, task)

        # Initialize process manager for background processes
        from llmling_agent.agent.process_manager import ProcessManager

        self.process_manager = ProcessManager()
        self.pool_talk = TeamTalk[Any].from_nodes(list(self.nodes.values()))
        if self.manifest.pool_server and self.manifest.pool_server.enabled:
            from llmling_agent.resource_providers.pool import PoolResourceProvider
            from llmling_agent_mcp.server import LLMLingServer

            provider = PoolResourceProvider(
                self, zed_mode=self.manifest.pool_server.zed_mode
            )
            self.server: LLMLingServer | None = LLMLingServer(
                provider=provider,
                config=self.manifest.pool_server,
            )
            self.progress_handlers.add_handler(self.server.report_progress)
        else:
            self.server = None
        # Create requested agents immediately
        for name in self.manifest.agents:
            agent = self.manifest.get_agent(name, deps=shared_deps)
            self.register(name, agent)

        # Then set up worker relationships
        for agent in self.agents.values():
            self.setup_agent_workers(agent)
        self._create_teams()
        # Set up forwarding connections
        if connect_nodes:
            self._connect_nodes()

    async def __aenter__(self) -> Self:
        """Enter async context and initialize all agents."""
        try:
            # Add MCP tool provider to all agents
            agents = list(self.agents.values())
            teams = list(self.teams.values())
            for agent in agents:
                agent.tools.add_provider(self.mcp)

            # Collect all components to initialize
            components: list[AbstractAsyncContextManager[Any]] = [
                self.mcp,
                *agents,
                *teams,
            ]

            # Add MCP server if configured
            if self.server:
                components.append(self.server)

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
            if self.mcp in agent.tools.providers:
                agent.tools.remove_provider(self.mcp)
        await self.cleanup()

    async def cleanup(self):
        """Clean up all agents."""
        # Clean up background processes first
        await self.process_manager.cleanup()
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

    async def run_event_loop(self):
        """Run pool in event-watching mode until interrupted."""
        print("Starting event watch mode...")
        print("Active nodes: ", ", ".join(self.list_nodes()))
        print("Press Ctrl+C to stop")

        with suppress(KeyboardInterrupt):
            while True:
                await asyncio.sleep(1)

    @property
    def agents(self) -> dict[str, AnyAgent[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        return {
            i.name: i
            for i in self._items.values()
            if isinstance(i, Agent | StructuredAgent)
        }

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
    def event_nodes(self) -> dict[str, EventNode[Any]]:
        """Get agents dict (backward compatibility)."""
        from llmling_agent.messaging.eventnode import EventNode

        return {i.name: i for i in self._items.values() if isinstance(i, EventNode)}

    @property
    def node_events(self) -> DictEvents:
        """Get node events."""
        return self._items.events

    @property
    def _error_class(self) -> type[LLMLingError]:
        """Error class for agent operations."""
        return LLMLingError

    def _validate_item(
        self, item: MessageEmitter[Any, Any] | Any
    ) -> MessageEmitter[Any, Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated Node

        Raises:
            LLMlingError: If item is not a valid node
        """
        if not isinstance(item, MessageEmitter):
            msg = f"Item must be Agent or Team, got {type(item)}"
            raise self._error_class(msg)
        item.context.pool = self
        return item

    def _create_teams(self):
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

    def _connect_nodes(self):
        """Set up connections defined in manifest."""
        # Merge agent and team configs into one dict of nodes with connections
        for name, config in self.manifest.nodes.items():
            source = self[name]
            for target in config.connections or []:
                match target:
                    case NodeConnectionConfig():
                        if target.name not in self:
                            msg = f"Forward target {target.name} not found for {name}"
                            raise ValueError(msg)
                        target_node = self[target.name]
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
            provider=new_config.get_provider(),
            system_prompt=new_config.system_prompts,
            name=new_name or f"{config.name}_copy_{len(self.agents)}",
        )
        if isinstance(original_agent, StructuredAgent):
            new_agent = new_agent.to_structured(original_agent.actual_type)

        # Register in pool
        agent_name = new_agent.name
        self.manifest.agents[agent_name] = new_config
        self.register(agent_name, new_agent)
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

    def setup_agent_workers(self, agent: AnyAgent[Any, Any]):
        """Set up workers for an agent from configuration."""
        for worker_config in agent.context.config.workers:
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
        from llmling_agent.agent.context import AgentContext

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

    def list_nodes(self) -> list[str]:
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
        result_type: type[TResult] | str | StructuredResponseConfig,
        **kwargs: Unpack[AgentKwargs],
    ) -> StructuredAgent[Any, TResult]: ...

    async def add_agent(
        self,
        name: AgentName,
        *,
        result_type: type[Any] | str | StructuredResponseConfig | None = None,
        **kwargs: Unpack[AgentKwargs],
    ) -> Agent[Any] | StructuredAgent[Any, Any]:
        """Add a new permanent agent to the pool.

        Args:
            name: Name for the new agent
            result_type: Optional type for structured responses:
                - None: Regular unstructured agent
                - type: Python type for validation
                - str: Name of response definition
                - StructuredResponseConfig: Complete response definition
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
                        if fn := talk.filter_condition:  # type: ignore
                            details.append(f"filter:{fn.__name__}")
                        if fn := talk.stop_condition:  # type: ignore
                            details.append(f"stop:{fn.__name__}")
                        if fn := talk.exit_condition:  # type: ignore
                            details.append(f"exit:{fn.__name__}")

                        label = f"|{' '.join(details)}|" if details else ""
                        lines.append(f"    {source}--{label}-->{target.name}")
                    else:
                        lines.append(f"    {source}-->{target.name}")

        return "\n".join(lines)


if __name__ == "__main__":

    async def main():
        path = "src/llmling_agent/config_resources/agents.yml"
        async with AgentPool[None](path) as pool:
            agent: Agent[None] = pool.get_agent("overseer")
            print(agent)

    import asyncio

    asyncio.run(main())
