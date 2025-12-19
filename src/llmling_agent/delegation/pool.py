"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from asyncio import Lock
from contextlib import AsyncExitStack, asynccontextmanager, suppress
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Unpack, overload

from anyenv import ProcessManager
from upathtools import UPath

from llmling_agent.agents import Agent
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


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from pydantic_ai.output import OutputSpec
    from tokonomics import ModelName
    from upathtools import JoinablePathLike

    from llmling_agent.agents.acp_agent import ACPAgent
    from llmling_agent.agents.agent import AgentKwargs
    from llmling_agent.agents.agui_agent import AGUIAgent
    from llmling_agent.agents.base_agent import BaseAgent
    from llmling_agent.common_types import (
        AgentName,
        BuiltinEventHandlerType,
        IndividualEventHandler,
        SessionIdType,
        SupportsStructuredOutput,
    )
    from llmling_agent.delegation.base_team import BaseTeam
    from llmling_agent.mcp_server.tool_bridge import ToolManagerBridge
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
        event_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
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
        from llmling_agent.sessions import SessionManager
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
        session_store = self.manifest.storage.get_session_store()
        self.sessions = SessionManager(pool=self, store=session_store)
        self.event_handlers = event_handlers or []
        self.connection_registry = ConnectionRegistry()
        servers = self.manifest.get_mcp_servers()
        self.mcp = MCPManager(name="pool_mcp", servers=servers, owner="pool")
        self.skills = SkillsManager(name="pool_skills", owner="pool")
        self._tool_bridges: dict[str, ToolManagerBridge] = {}
        self._tasks = TaskRegistry()
        # Register tasks from manifest
        for name, task in self.manifest.jobs.items():
            self._tasks.register(name, task)
        self.process_manager = ProcessManager()
        self.pool_talk = TeamTalk[Any].from_nodes(list(self.nodes.values()))
        # Create requested agents immediately
        for name in self.manifest.agents:
            agent = self.create_agent(
                name,
                deps_type=shared_deps_type,
                input_provider=self._input_provider,
                pool=self,
                event_handlers=self.event_handlers,
            )
            self.register(name, agent)

        # Create ACP agents
        for name in self.manifest.acp_agents:
            acp_agent = self.create_acp_agent(name, deps_type=shared_deps_type)
            acp_agent.agent_pool = self
            self.register(name, acp_agent)

        # Create AG-UI agents
        for name in self.manifest.agui_agents:
            agui_agent = self.create_agui_agent(name, deps_type=shared_deps_type)
            agui_agent.agent_pool = self
            self.register(name, agui_agent)

        self._create_teams()
        if connect_nodes:
            self._connect_nodes()

        self._enter_lock = Lock()  # Initialize async safety fields
        self._running_count = 0

    async def __aenter__(self) -> Self:
        """Enter async context and initialize all agents."""
        if self._running_count > 0:
            self._running_count += 1
            return self
        async with self._enter_lock:
            try:
                # Initialize MCP manager first, then add aggregating provider
                await self.exit_stack.enter_async_context(self.mcp)
                await self.exit_stack.enter_async_context(self.skills)
                aggregating_provider = self.mcp.get_aggregating_provider()
                agents = list(self.agents.values())
                acp_agents = list(self.acp_agents.values())
                agui_agents = list(self.agui_agents.values())
                teams = list(self.teams.values())
                for agent in agents:
                    agent.tools.add_provider(aggregating_provider)
                # Collect remaining components to initialize (MCP already initialized)
                components: list[AbstractAsyncContextManager[Any]] = [
                    self.storage,
                    self.sessions,
                    *agents,
                    *acp_agents,
                    *agui_agents,
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
                for agent in self.agents.values():
                    agent.tools.remove_provider(aggregating_provider.name)
                await self.cleanup()

    async def create_tool_bridge(
        self,
        node: BaseAgent[Any, Any],
        *,
        name: str = "pool_tools",
        host: str = "127.0.0.1",
        port: int = 0,
        transport: str = "sse",
    ) -> ToolManagerBridge:
        """Create and start a tool bridge for exposing tools to ACP agents.

        This creates an in-process MCP server that exposes the given node's
        tools. The returned bridge can be added to ACP agents to give them access
        to internal toolsets.

        Args:
            node: The agent node whose tools to expose
            name: Unique name for this bridge
            host: Host to bind the HTTP server to
            port: Port to bind to (0 = auto-select)
            transport: Transport protocol ('sse' or 'streamable-http')

        Returns:
            Started ToolManagerBridge instance

        Example:
            ```python
            async with AgentPool() as pool:
                # Create bridge from an agent's tools
                bridge = await pool.create_tool_bridge(
                    pool.agents["orchestrator"],
                    name="orchestrator_tools",
                )
                # Add to ACP agent
                await pool.acp_agents["claude"].add_tool_bridge(bridge)
            ```
        """
        from llmling_agent.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge

        if name in self._tool_bridges:
            msg = f"Tool bridge {name!r} already exists"
            raise ValueError(msg)

        config = BridgeConfig(
            host=host,
            port=port,
            transport=transport,
            server_name=f"llmling-{name}",
        )
        bridge = ToolManagerBridge(node=node, config=config)
        await bridge.start()
        self._tool_bridges[name] = bridge
        return bridge

    async def get_tool_bridge(self, name: str) -> ToolManagerBridge:
        """Get a tool bridge by name."""
        if name not in self._tool_bridges:
            msg = f"Tool bridge {name!r} not found"
            raise KeyError(msg)
        return self._tool_bridges[name]

    async def remove_tool_bridge(self, name: str) -> None:
        """Stop and remove a tool bridge."""
        if name in self._tool_bridges:
            await self._tool_bridges[name].stop()
            del self._tool_bridges[name]

    @property
    def is_running(self) -> bool:
        """Check if the agent pool is running."""
        return bool(self._running_count)

    async def cleanup(self) -> None:
        """Clean up all agents."""
        # Clean up tool bridges first
        for bridge in list(self._tool_bridges.values()):
            await bridge.stop()
        self._tool_bridges.clear()
        # Clean up background processes
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
        picker: SupportsStructuredOutput | None = None,
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
        picker: SupportsStructuredOutput | None = None,
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
        picker: SupportsStructuredOutput | None = None,
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
        picker: SupportsStructuredOutput | None = None,
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
        team = TeamRun(
            [self.get_agent(i) if isinstance(i, str) else i for i in agents],
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
        picker: SupportsStructuredOutput | None = None,
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
        picker: SupportsStructuredOutput | None = None,
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
        picker: SupportsStructuredOutput | None = None,
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

        team = Team(
            name=name,
            description=description,
            agents=[self.get_agent(i) if isinstance(i, str) else i for i in agents],
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
        print("Active nodes: ", ", ".join(list(self.nodes.keys())))
        print("Press Ctrl+C to stop")

        shutdown_event = asyncio.Event()
        with suppress(KeyboardInterrupt):
            await shutdown_event.wait()

    @property
    def agents(self) -> dict[str, Agent[Any, Any]]:
        """Get regular agents dict."""
        return {i.name: i for i in self._items.values() if isinstance(i, Agent)}

    @property
    def acp_agents(self) -> dict[str, ACPAgent]:
        """Get ACP agents dict."""
        from llmling_agent.agents.acp_agent import ACPAgent

        return {i.name: i for i in self._items.values() if isinstance(i, ACPAgent)}

    @property
    def agui_agents(self) -> dict[str, AGUIAgent]:
        """Get AG-UI agents dict."""
        from llmling_agent.agents.agui_agent import AGUIAgent

        return {i.name: i for i in self._items.values() if isinstance(i, AGUIAgent)}

    @property
    def all_agents(self) -> dict[str, BaseAgent[Any, Any]]:
        """Get all agents (regular, ACP, and AG-UI)."""
        from llmling_agent.agents.base_agent import BaseAgent

        return {i.name: i for i in self._items.values() if isinstance(i, BaseAgent)}

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
            team.nodes.extend(members)
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
                        agent_name = f"file_writer_{Path(path_obj).stem}"
                        target_node = Agent(model=target.get_model(), name=agent_name)
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
        from llmling_agent.agents import Agent

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
        from llmling_agent.agents import Agent

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

    def create_agent[TAgentDeps](
        self,
        name: str,
        deps_type: type[TAgentDeps] | None = None,
        input_provider: InputProvider | None = None,
        pool: AgentPool[Any] | None = None,
        event_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> Agent[TAgentDeps, Any]:
        from llmling_agent import Agent
        from llmling_agent.utils.result_utils import to_type
        from llmling_agent_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        manifest = self.manifest
        # Get config from inline agents or file agents
        if name in manifest.agents:
            config = manifest.agents[name]
        elif name in manifest.file_agents:
            config = manifest._loaded_file_agents[name]
        else:
            msg = f"Agent {name!r} not found in agents or file_agents"
            raise KeyError(msg)
        sys_prompts: list[str] = []
        for prompt in config.system_prompts:
            match prompt:
                case (str() as sys_prompt) | StaticPromptConfig(content=sys_prompt):
                    sys_prompts.append(sys_prompt)
                case FilePromptConfig(path=path, variables=variables):
                    template_path = Path(path)  # Load template from file
                    if not template_path.is_absolute() and config.config_file_path:
                        template_path = Path(config.config_file_path).parent / path

                    template_content = template_path.read_text("utf-8")
                    if variables:  # Apply variables if any
                        from jinja2 import Template

                        template = Template(template_content)
                        content = template.render(**variables)
                    else:
                        content = template_content
                    sys_prompts.append(content)
                case LibraryPromptConfig(reference=reference):
                    try:  # Load from library
                        content = self.manifest.prompt_manager.get.sync(reference)
                        sys_prompts.append(content)
                    except Exception as e:
                        msg = f"Failed to load library prompt {reference!r} for agent {name}"
                        logger.exception(msg)
                        raise ValueError(msg) from e
                case FunctionPromptConfig(function=function, arguments=arguments):
                    content = function(**arguments)  # Call function to get prompt content
                    sys_prompts.append(content)
        # Prepare toolsets list with config's tool provider
        toolsets_list = config.get_toolsets()
        if config_tool_provider := config.get_tool_provider():
            toolsets_list.append(config_tool_provider)
        # Convert workers config to a toolset (backwards compatibility)
        if config.workers:
            from llmling_agent_toolsets.builtin.workers import WorkersTools

            workers_provider = WorkersTools(workers=list(config.workers), name="workers")
            toolsets_list.append(workers_provider)
        # Step 1: Get agent-specific output type (same as before)
        agent_output_type = manifest.get_output_type(name) or str
        # Step 2: Resolve it fully with to_type (same as before)
        resolved_output_type = to_type(agent_output_type, manifest.responses)
        # Merge pool-level handlers with config-level handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]
        return Agent(
            # context=context,
            model=config.model
            if isinstance(config.model, str) or config.model is None
            else config.model.get_model(),
            system_prompt=sys_prompts,
            name=name,
            display_name=config.display_name,
            deps_type=deps_type,
            env=config.environment.get_provider() if config.environment else None,
            description=config.description,
            retries=config.retries,
            session=config.get_session_config(),
            output_retries=config.output_retries,
            end_strategy=config.end_strategy,
            debug=config.debug,
            agent_config=config,
            input_provider=input_provider,
            output_type=resolved_output_type,
            event_handlers=merged_handlers or None,
            agent_pool=pool,
            tool_mode=config.tool_mode,
            knowledge=config.knowledge,
            toolsets=toolsets_list,
            auto_cache=config.auto_cache,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
            tool_confirmation_mode=config.requires_tool_confirmation,
        )

    def create_acp_agent[TDeps](
        self,
        name: str,
        deps_type: type[TDeps] | None = None,
    ) -> ACPAgent[TDeps]:
        """Create an ACPAgent from configuration.

        Args:
            name: Name of the ACP agent in the manifest
            deps_type: Optional dependency type (not used by ACP agents currently)

        Returns:
            Configured ACPAgent instance
        """
        from llmling_agent.agents.acp_agent import ACPAgent

        config = self.manifest.acp_agents[name]
        # Ensure name is set on config
        if config.name is None:
            config = config.model_copy(update={"name": name})
        # Merge pool-level handlers with config-level handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *self.event_handlers,
        ]
        return ACPAgent(config=config, event_handlers=merged_handlers or None)

    def create_agui_agent[TDeps](
        self,
        name: str,
        deps_type: type[TDeps] | None = None,
    ) -> AGUIAgent[TDeps]:
        """Create an AGUIAgent from configuration.

        Args:
            name: Name of the AG-UI agent in the manifest
            deps_type: Optional dependency type (not used by AG-UI agents currently)

        Returns:
            Configured AGUIAgent instance
        """
        from llmling_agent.agents.agui_agent import AGUIAgent

        config = self.manifest.agui_agents[name]
        # Ensure name is set on config
        if config.name is None:
            config = config.model_copy(update={"name": name})
        # Merge pool-level handlers with config-level handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *self.event_handlers,
        ]
        return AGUIAgent(
            endpoint=config.endpoint,
            name=config.name or "agui-agent",
            description=config.description,
            display_name=config.display_name,
            event_handlers=merged_handlers or None,
            timeout=config.timeout,
            headers=config.headers,
            startup_command=config.startup_command,
            startup_delay=config.startup_delay,
            tools=[tool_config.get_tool() for tool_config in config.tools],
            mcp_servers=config.mcp_servers,
            tool_confirmation_mode=config.requires_tool_confirmation,
        )


if __name__ == "__main__":

    async def main() -> None:
        path = "src/llmling_agent/config_resources/agents.yml"
        async with AgentPool(path) as pool:
            agent = pool.get_agent("overseer")
            print(agent)

    asyncio.run(main())
