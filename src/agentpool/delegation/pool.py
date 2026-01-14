"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from asyncio import Lock
from contextlib import AsyncExitStack, asynccontextmanager, suppress
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Unpack, overload

from anyenv import ProcessManager
import anyio
from upathtools import UPath

from agentpool.agents import Agent
from agentpool.common_types import NodeName
from agentpool.delegation.message_flow_tracker import MessageFlowTracker
from agentpool.delegation.team import Team
from agentpool.delegation.teamrun import TeamRun
from agentpool.log import get_logger
from agentpool.messaging import MessageNode
from agentpool.talk import TeamTalk
from agentpool.talk.registry import ConnectionRegistry
from agentpool.tasks import TaskRegistry
from agentpool.utils.baseregistry import BaseRegistry
from agentpool_config.forward_targets import (
    CallableConnectionConfig,
    FileConnectionConfig,
    NodeConnectionConfig,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model
    from pydantic_ai.output import OutputSpec
    from upathtools import JoinablePathLike

    from agentpool.agents.acp_agent import ACPAgent
    from agentpool.agents.agent import AgentKwargs
    from agentpool.agents.agui_agent import AGUIAgent
    from agentpool.agents.base_agent import BaseAgent
    from agentpool.agents.claude_code_agent import ClaudeCodeAgent
    from agentpool.common_types import (
        AgentName,
        BuiltinEventHandlerType,
        IndividualEventHandler,
    )
    from agentpool.delegation.base_team import BaseTeam
    from agentpool.mcp_server.tool_bridge import ToolManagerBridge
    from agentpool.messaging import ChatMessage
    from agentpool.messaging.compaction import CompactionPipeline
    from agentpool.models import AnyAgentConfig
    from agentpool.models.manifest import AgentsManifest
    from agentpool.ui.base import InputProvider
    from agentpool_config.task import Job


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
        from agentpool.mcp_server.manager import MCPManager
        from agentpool.models.manifest import AgentsManifest
        from agentpool.observability import registry
        from agentpool.sessions import SessionManager
        from agentpool.skills.manager import SkillsManager
        from agentpool.storage import StorageManager
        from agentpool.utils.streams import FileOpsTracker, TodoTracker
        from agentpool_toolsets.builtin.debug import install_memory_handler

        super().__init__()

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
        self._memory_log_handler = install_memory_handler()
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
        self.file_ops = FileOpsTracker()
        # Todo/plan tracker for task management
        self.todos = TodoTracker()
        # Create all agents from unified manifest.agents dict
        for name, config in self.manifest.agents.items():
            agent = self._create_agent_from_config(name, config, deps_type=shared_deps_type)
            agent.agent_pool = self
            self.register(name, agent)

        self._create_teams()
        if connect_nodes:
            self._connect_nodes()
        self.pool_talk = TeamTalk[Any].from_nodes(list(self.nodes.values()))
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
                agents = list(self.all_agents.values())
                teams = list(self.teams.values())
                for agent in agents:
                    agent.tools.add_provider(aggregating_provider)
                # Collect remaining components to initialize (MCP already initialized)
                comps: list[AbstractAsyncContextManager[Any]] = [
                    self.storage,
                    self.sessions,
                    *agents,
                    *teams,
                ]
                node_inits = [self.exit_stack.enter_async_context(c) for c in comps]
                if self.parallel_load:
                    await asyncio.gather(*node_inits)
                else:
                    for init in node_inits:
                        await init

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
    ) -> ToolManagerBridge:
        """Create and start a tool bridge for exposing tools to external agents.

        This creates an in-process MCP server that exposes the given node's
        tools. The returned bridge can be added to ACP agents or Claude Code
        agents to give them access to internal toolsets.

        Args:
            node: The agent node whose tools to expose
            name: Unique name for this bridge
            host: Host to bind the HTTP server to
            port: Port to bind to (0 = auto-select)

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
                # Or add to Claude Code agent
                await pool.claude_code_agents["coder"].add_tool_bridge(bridge)
            ```
        """
        from agentpool.mcp_server.tool_bridge import BridgeConfig, ToolManagerBridge

        if name in self._tool_bridges:
            msg = f"Tool bridge {name!r} already exists"
            raise ValueError(msg)

        config = BridgeConfig(
            host=host,
            port=port,
            server_name=f"agentpool-{name}",
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
    ) -> TeamRun[Any, TResult]: ...

    def create_team_run[TResult](
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]] | None = None,
        validator: MessageNode[Any, TResult] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
    ) -> TeamRun[Any, TResult]:
        """Create a a sequential TeamRun from a list of Agents.

        Args:
            agents: List of agent names or team/agent instances (all if None)
            validator: Node to validate the results of the TeamRun
            name: Optional name for the team
            description: Optional description for the team
            shared_prompt: Optional prompt for all agents
        """
        from agentpool.delegation.teamrun import TeamRun

        if agents is None:
            agents = list(self.agents.keys())
        team = TeamRun(
            [self.get_agent(i) if isinstance(i, str) else i for i in agents],
            name=name,
            description=description,
            validator=validator,
            shared_prompt=shared_prompt,
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
    ) -> Team[TDeps]: ...

    @overload
    def create_team(
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
    ) -> Team[Any]: ...

    def create_team(
        self,
        agents: Sequence[AgentName | MessageNode[Any, Any]] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
    ) -> Team[Any]:
        """Create a group from agent names or instances.

        Args:
            agents: List of agent names or instances (all if None)
            name: Optional name for the team
            description: Optional description for the team
            shared_prompt: Optional prompt for all agents
        """
        from agentpool.delegation.team import Team

        if agents is None:
            agents = list(self.agents.keys())

        team = Team(
            name=name,
            description=description,
            agents=[self.get_agent(i) if isinstance(i, str) else i for i in agents],
            shared_prompt=shared_prompt,
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

        shutdown_event = anyio.Event()
        with suppress(KeyboardInterrupt):
            await shutdown_event.wait()

    @property
    def agents(self) -> dict[str, Agent[Any, Any]]:
        """Get regular agents dict."""
        return {i.name: i for i in self._items.values() if isinstance(i, Agent)}

    @property
    def acp_agents(self) -> dict[str, ACPAgent]:
        """Get ACP agents dict."""
        from agentpool.agents.acp_agent import ACPAgent

        return {i.name: i for i in self._items.values() if isinstance(i, ACPAgent)}

    @property
    def agui_agents(self) -> dict[str, AGUIAgent]:
        """Get AG-UI agents dict."""
        from agentpool.agents.agui_agent import AGUIAgent

        return {i.name: i for i in self._items.values() if isinstance(i, AGUIAgent)}

    @property
    def claude_code_agents(self) -> dict[str, ClaudeCodeAgent]:
        """Get Claude Code agents dict."""
        from agentpool.agents.claude_code_agent import ClaudeCodeAgent

        return {i.name: i for i in self._items.values() if isinstance(i, ClaudeCodeAgent)}

    @property
    def all_agents(self) -> dict[str, BaseAgent[Any, Any]]:
        """Get all agents (regular, ACP, and AG-UI)."""
        from agentpool.agents.base_agent import BaseAgent

        return {i.name: i for i in self._items.values() if isinstance(i, BaseAgent)}

    @property
    def teams(self) -> dict[str, BaseTeam[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        from agentpool.delegation.base_team import BaseTeam

        return {i.name: i for i in self._items.values() if isinstance(i, BaseTeam)}

    @property
    def nodes(self) -> dict[str, MessageNode[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        from agentpool import MessageNode

        return {i.name: i for i in self._items.values() if isinstance(i, MessageNode)}

    @property
    def compaction_pipeline(self) -> CompactionPipeline | None:
        """Get the configured compaction pipeline for message history management.

        Returns:
            CompactionPipeline instance or None if not configured
        """
        return self.manifest.get_compaction_pipeline()

    def _validate_item(self, item: MessageNode[Any, Any] | Any) -> MessageNode[Any, Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated Node

        Raises:
            AgentPoolError: If item is not a valid node
        """
        if not isinstance(item, MessageNode):
            msg = f"Item must be Agent or Team, got {type(item)}"
            raise self._error_class(msg)
        item.agent_pool = self
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
                source.connections.set_wait_state(target_node, wait=target.wait_for_completion)

    @overload
    def get_agent[TResult = str](
        self,
        agent: AgentName | Agent[Any, str],
        *,
        output_type: type[TResult] = str,  # type: ignore
    ) -> Agent[TPoolDeps, TResult]: ...

    @overload
    def get_agent[TCustomDeps, TResult = str](
        self,
        agent: AgentName | Agent[Any, str],
        *,
        deps_type: type[TCustomDeps],
        output_type: type[TResult] = str,  # type: ignore
    ) -> Agent[TCustomDeps, TResult]: ...

    def get_agent(
        self,
        agent: AgentName | Agent[Any, str],
        *,
        deps_type: Any | None = None,
        output_type: Any = str,
    ) -> Agent[Any, Any]:
        """Get or configure an agent from the pool.

        This method provides flexible agent configuration with dependency injection:
        - Without deps: Agent uses pool's shared dependencies
        - With deps: Agent uses provided custom dependencies

        Args:
            agent: Either agent name or instance
            deps_type: Optional custom dependencies type (overrides shared deps)
            output_type: Optional type for structured responses

        Returns:
            Either:
            - Agent[TPoolDeps] when using pool's shared deps
            - Agent[TCustomDeps] when custom deps provided

        Raises:
            KeyError: If agent name not found
            ValueError: If configuration is invalid

        Note:
            To change the model, call `await agent.set_model(model_id)` after
            getting the agent.
        """
        from agentpool.agents import Agent

        base = agent if isinstance(agent, Agent) else self.agents[agent]
        # Use custom deps if provided, otherwise use shared deps
        # base.context.data = deps if deps is not None else self.shared_deps
        base.deps_type = deps_type
        base.agent_pool = self
        if output_type not in {str, None}:
            base.to_structured(output_type)
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
        from agentpool.agents import Agent

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

    def get_message_chain(self, message: ChatMessage[Any]) -> list[str]:
        """Get the chain of agent names that processed a message.

        Reconstructs the forwarding chain by walking parent_id references
        across all agents' conversations in the pool.

        Args:
            message: The message to trace back

        Returns:
            List of agent names in order from origin to current message's agent.
            Empty list if message has no parent or parent not found.
        """
        # Build index of all messages by ID across all agents
        message_index: dict[str, tuple[ChatMessage[Any], str]] = {}
        for agent in self.agents.values():
            for msg in agent.conversation.chat_messages:
                message_index[msg.message_id] = (msg, agent.name)

        # Walk back through parent_id chain
        chain: list[str] = []
        current_id = message.parent_id

        while current_id and current_id in message_index:
            parent_msg, agent_name = message_index[current_id]
            # Only add agent name if it's different from previous (avoid duplicates)
            if not chain or chain[-1] != agent_name:
                chain.append(agent_name)
            current_id = parent_msg.parent_id

        # Reverse to get origin -> current order
        chain.reverse()
        return chain

    def find_message_by_id(self, message_id: str) -> ChatMessage[Any] | None:
        """Find a message by ID across all agents in the pool.

        Args:
            message_id: The message ID to search for

        Returns:
            The ChatMessage if found, None otherwise
        """
        for agent in self.agents.values():
            for msg in agent.conversation.chat_messages:
                if msg.message_id == message_id:
                    return msg
        return None

    def _create_agent_from_config[TAgentDeps](
        self,
        name: str,
        config: AnyAgentConfig,
        deps_type: type[TAgentDeps] | None = None,
    ) -> BaseAgent[TAgentDeps, Any]:
        """Create an agent from a unified config, dispatching by type.

        Args:
            name: Agent name
            config: Agent configuration (any type)
            deps_type: Optional dependency type

        Returns:
            Configured agent instance
        """
        from agentpool.models.acp_agents import BaseACPAgentConfig
        from agentpool.models.agents import NativeAgentConfig
        from agentpool.models.agui_agents import AGUIAgentConfig
        from agentpool.models.claude_code_agents import ClaudeCodeAgentConfig

        # Ensure name is set on config
        if config.name is None:
            config = config.model_copy(update={"name": name})

        match config:
            case NativeAgentConfig():
                from agentpool import Agent

                return Agent.from_config(
                    config,
                    name=name,  # Pass dict key for manifest lookups
                    manifest=self.manifest,
                    event_handlers=self.event_handlers,
                    input_provider=self._input_provider,
                    agent_pool=self,
                    deps_type=deps_type,
                )
            case AGUIAgentConfig():
                from agentpool.agents.agui_agent import AGUIAgent

                return AGUIAgent.from_config(
                    config,
                    event_handlers=self.event_handlers,
                    agent_pool=self,
                )
            case ClaudeCodeAgentConfig():
                from agentpool.agents.claude_code_agent import ClaudeCodeAgent
                from agentpool.utils.result_utils import to_type

                output_type: type | None = None
                if config.output_type:
                    output_type = to_type(config.output_type, self.manifest.responses)
                return ClaudeCodeAgent.from_config(
                    config,
                    event_handlers=self.event_handlers,
                    input_provider=self._input_provider,
                    agent_pool=self,
                    output_type=output_type,
                )
            case BaseACPAgentConfig():
                from agentpool.agents.acp_agent import ACPAgent

                return ACPAgent.from_config(
                    config,
                    event_handlers=self.event_handlers,
                    agent_pool=self,
                )
            case _:
                msg = f"Unknown agent config type: {type(config)}"
                raise TypeError(msg)

    def create_agent[TAgentDeps](  # noqa: PLR0915
        self,
        name: str,
        deps_type: type[TAgentDeps] | None = None,
        input_provider: InputProvider | None = None,
        event_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> Agent[TAgentDeps, Any]:
        from agentpool import Agent
        from agentpool.models.agents import NativeAgentConfig
        from agentpool.utils.result_utils import to_type
        from agentpool_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        manifest = self.manifest
        # Get config from inline agents or file agents
        if name in manifest.agents:
            config = manifest.agents[name]
            if not isinstance(config, NativeAgentConfig):
                msg = f"Agent {name!r} is not a native agent, use appropriate create method"
                raise TypeError(msg)
        elif name in manifest.file_agents:
            config = manifest._loaded_file_agents[name]
        else:
            msg = f"Agent {name!r} not found in agents or file_agents"
            raise KeyError(msg)
        # Normalize system_prompt to a list for iteration
        sys_prompts: list[str] = []
        prompt_source = config.system_prompt
        if prompt_source is not None:
            prompts_to_process = (
                [prompt_source] if isinstance(prompt_source, str) else prompt_source
            )
            for prompt in prompts_to_process:
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
            from agentpool_toolsets.builtin.workers import WorkersTools

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
        # Resolve model (handles variants and wraps strings)
        resolved_model = self.manifest.resolve_model(config.model)
        model: Model | str = resolved_model.get_model()
        model_settings: ModelSettings | None = resolved_model.get_model_settings()
        # Extract pydantic-ai builtin tools from config
        builtin_tools = config.get_builtin_tools()

        return Agent(
            # context=context,
            model=model,
            model_settings=model_settings,
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
            agent_config=config,
            input_provider=input_provider,
            output_type=resolved_output_type,
            event_handlers=merged_handlers or None,
            agent_pool=self,
            tool_mode=config.tool_mode,
            knowledge=config.knowledge,
            toolsets=toolsets_list,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
            tool_confirmation_mode=config.requires_tool_confirmation,
            builtin_tools=builtin_tools or None,
            usage_limits=config.usage_limits,
            providers=config.model_providers,
        )


if __name__ == "__main__":

    async def main() -> None:
        path = "src/agentpool/config_resources/agents.yml"
        async with AgentPool(path) as pool:
            agent = pool.get_agent("overseer")
            print(agent)

    anyio.run(main)
