"""Provider for agent management tools."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llmling import ToolError
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent

logger = get_logger(__name__)


async def delegate_to(  # noqa: D417
    ctx: AgentContext,
    agent_or_team_name: str,
    prompt: str,
) -> str:
    """Delegate a task to an agent or team.

    If an action requires you to delegate a task, this tool can be used to assign and
    execute a task. Instructions can be passed via the prompt parameter.

    Args:
        agent_or_team_name: The agent or team to delegate the task to
        prompt: Instructions for the agent or team to delegate to.

    Returns:
        The result of the delegated task
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "Agent needs to be in a pool to delegate tasks"
        raise ToolError(msg)

    if agent_or_team_name in ctx.pool.nodes:
        node = ctx.pool.nodes[agent_or_team_name]
        result = await node.run(prompt)
        return result.format(style="detailed", show_costs=True)

    msg = f"No agent or team found with name: {agent_or_team_name}"
    raise ToolError(msg)


async def list_available_agents(  # noqa: D417
    ctx: AgentContext,
    only_idle: bool = False,
    detailed: bool = False,
) -> str:
    """List all agents available in the current pool.

    Args:
        only_idle: If True, only returns agents that aren't currently busy.
                    Use this to find agents ready for immediate tasks.
        detailed: If True, additional info for each team is provided (e.g. description)

    Returns:
        List of agent names that you can use with delegate_to
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    agents = dict(ctx.pool.agents)
    if only_idle:
        agents = {name: agent for name, agent in agents.items() if not agent.is_busy()}
    if not detailed:
        return "\n".join(agents.keys())
    lines = []
    for name, agent in agents.items():
        lines.extend([
            f"name: {name}",
            f"description: {agent.description or 'No description'}",
            f"model: {agent.model_name}",
            "---",
        ])

    return "\n".join(lines) if lines else "No agents available"


async def list_available_teams(  # noqa: D417
    ctx: AgentContext,
    only_idle: bool = False,
    detailed: bool = False,
) -> str:
    """List all available teams in the pool.

    Args:
        only_idle: If True, only returns teams that aren't currently executing
        detailed: If True, additional info for each team is provided (e.g. description)

    Returns:
        Formatted list of teams with their descriptions and types
    """
    from llmling_agent import TeamRun

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    teams = ctx.pool.teams
    if only_idle:
        teams = {name: team for name, team in teams.items() if not team.is_running}
    if not detailed:
        return "\n".join(teams.keys())
    lines = []
    for name, team in teams.items():
        lines.extend([
            f"name: {name}",
            f"description: {team.description or 'No description'}",
            f"type: {'sequential' if isinstance(team, TeamRun) else 'parallel'}",
            "members: " + ", ".join(a.name for a in team.agents),
            "---",
        ])

    return "\n".join(lines) if lines else "No teams available"


async def create_worker_agent[TDeps](
    ctx: AgentContext[TDeps],
    name: str,
    system_prompt: str,
    model: str | None = None,
) -> str:
    """Create a new agent and register it as a tool.

    The new agent will be available as a tool for delegating specific tasks.
    It inherits the current model unless overridden.
    """
    from llmling_agent import Agent

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    model = model or ctx.agent.model_name
    agent = Agent[TDeps](name=name, model=model, system_prompt=system_prompt, context=ctx)
    assert ctx.agent
    tool_info = ctx.agent.register_worker(agent)
    return f"Created worker agent and registered as tool: {tool_info.name}"


async def spawn_delegate[TDeps](
    ctx: AgentContext[TDeps],
    task: str,
    system_prompt: str,
    model: str | None = None,
    connect_back: bool = False,
) -> str:
    """Spawn a temporary agent for a specific task.

    Creates an ephemeral agent that will execute the task and clean up automatically
    Optionally connects back to receive results.
    """
    from llmling_agent import Agent

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    name = f"delegate_{uuid4().hex[:8]}"
    model = model or ctx.agent.model_name
    agent = Agent[TDeps](name=name, model=model, system_prompt=system_prompt, context=ctx)

    if connect_back:
        assert ctx.agent
        ctx.agent.connect_to(agent)
    try:
        await agent.run(task)
    except Exception as e:
        msg = f"Failed to spawn delegate {name}: {e}"
        raise ModelRetry(msg) from e
    return f"Spawned delegate {name} for task"


async def add_agent(  # noqa: D417
    ctx: AgentContext,
    name: str,
    system_prompt: str,
    model: str | None = None,
    tools: list[str] | None = None,
    session: str | None = None,
    result_type: str | None = None,
) -> str:
    """Add a new agent to the pool.

    Args:
        name: Name for the new agent
        system_prompt: System prompt defining agent's role/behavior
        model: Optional model override (uses default if not specified)
        tools: Imort paths of the tools the agent should have, if any.
        session: Session ID to recover conversation state from
        result_type: Name of response type from manifest (for structured output)

    Returns:
        Confirmation message about the created agent
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    assert ctx.pool, "No agent pool available"
    try:
        agent: AnyAgent[Any, Any] = await ctx.pool.add_agent(
            name=name,
            system_prompt=system_prompt,
            model=model,
            tools=tools,
            result_type=result_type,
            session=session,
        )
    except ValueError as e:  # for wrong tool imports
        raise ModelRetry(message=f"Error creating agent: {e}") from None
    return f"Created agent {agent.name} using model {agent.model_name}"


async def add_team(  # noqa: D417
    ctx: AgentContext,
    agents: list[str],
    mode: Literal["sequential", "parallel"] = "sequential",
    name: str | None = None,
) -> str:
    """Create a team from existing agents.

    Args:
        agents: Names of agents to include in team
        mode: How the team should operate:
            - sequential: Agents process in sequence (pipeline)
            - parallel: Agents process simultaneously
        name: Optional name for the team
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    # Verify all agents exist
    for agent_name in agents:
        if agent_name not in ctx.pool.agents:
            msg = f"Agent not found: {agent_name}"
            raise ToolError(msg)
    if mode == "sequential":
        ctx.pool.create_team_run(agents, name=name)
    else:
        ctx.pool.create_team(agents, name=name)
    mode_str = "pipeline" if mode == "sequential" else "parallel"
    return f"Created {mode_str} team with agents: {', '.join(agents)}"


async def ask_agent(  # noqa: D417
    ctx: AgentContext,
    agent_name: str,
    message: str,
    *,
    model: str | None = None,
    store_history: bool = True,
) -> str:
    """Send a message to a specific agent and get their response.

    Args:
        agent_name: Name of the agent to interact with
        message: Message to send to the agent
        model: Optional temporary model override
        store_history: Whether to store this exchange in history

    Returns:
        The agent's response
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    assert ctx.pool, "No agent pool available"
    agent = ctx.pool.get_agent(agent_name)
    try:
        result = await agent.run(message, model=model, store_history=store_history)
    except Exception as e:
        msg = f"Failed to ask agent {agent_name}: {e}"
        raise ModelRetry(msg) from e
    return str(result.content)


async def connect_nodes(  # noqa: D417
    ctx: AgentContext,
    source: str,
    target: str,
    *,
    connection_type: Literal["run", "context", "forward"] = "run",
    priority: int = 0,
    delay_seconds: float | None = None,
    queued: bool = False,
    queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
    wait_for_completion: bool = True,
    name: str | None = None,
) -> str:
    """Connect two nodes to enable message flow between them.

    Nodes can be agents, teams, or EventNodes.

    Args:
        source: Name of the source node
        target: Name of the target node
        connection_type: How messages should be handled:
            - run: Execute message as a new run in target
            - context: Add message as context to target
            - forward: Forward message to target's outbox
        priority: Task priority (lower = higher priority)
        delay_seconds: Optional delay before processing messages
        queued: Whether messages should be queued for manual processing
        queue_strategy: How to process queued messages:
            - concat: Combine all messages with newlines
            - latest: Use only the most recent message
            - buffer: Process all messages individually
        wait_for_completion: Whether to wait for target to complete
        name: Optional name for this connection

    Returns:
        Description of the created connection
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    # Get the nodes
    if source not in ctx.pool.nodes:
        msg = f"Source node not found: {source}"
        raise ToolError(msg)
    if target not in ctx.pool.nodes:
        msg = f"Target node not found: {target}"
        raise ToolError(msg)

    source_node = ctx.pool.nodes[source]
    target_node = ctx.pool.nodes[target]

    # Create the connection
    delay = timedelta(seconds=delay_seconds) if delay_seconds is not None else None
    _talk = source_node.connect_to(
        target_node,
        connection_type=connection_type,
        priority=priority,
        delay=delay,
        queued=queued,
        queue_strategy=queue_strategy,
        name=name,
    )
    source_node.connections.set_wait_state(target_node, wait=wait_for_completion)

    return (
        f"Created connection from {source} to {target} "
        f"(type={connection_type}, queued={queued}, "
        f"strategy={queue_strategy if queued else 'n/a'})"
    )


def create_agent_management_tools() -> list[Tool]:
    """Create tools for agent and team management operations."""
    return [
        Tool.from_callable(delegate_to, source="builtin", category="other"),
        Tool.from_callable(list_available_agents, source="builtin", category="search"),
        Tool.from_callable(list_available_teams, source="builtin", category="search"),
        Tool.from_callable(create_worker_agent, source="builtin", category="other"),
        Tool.from_callable(spawn_delegate, source="builtin", category="other"),
        Tool.from_callable(add_agent, source="builtin", category="other"),
        Tool.from_callable(add_team, source="builtin", category="other"),
        Tool.from_callable(ask_agent, source="builtin", category="other"),
        Tool.from_callable(connect_nodes, source="builtin", category="other"),
    ]


class AgentManagementTools(StaticResourceProvider):
    """Provider for agent management tools."""

    def __init__(self, name: str = "agent_management"):
        super().__init__(name=name, tools=create_agent_management_tools())


if __name__ == "__main__":
    # import logging
    from llmling_agent import AgentPool

    user_prompt = """Add a stdio MCP server:
// 	"command": "npx",
// 	"args": ["mcp-graphql"],
// 	"env": { "ENDPOINT": "https://diego.one/graphql" }

."""

    async def main():
        from llmling_agent_config.toolsets import IntegrationToolsetConfig

        async with AgentPool() as pool:
            toolsets = [IntegrationToolsetConfig()]
            toolset_providers = [config.get_provider() for config in toolsets]
            agent = await pool.add_agent(
                "X",
                toolsets=toolset_providers,
                model="openai:gpt-5-nano",
            )
            agent.tool_used.connect(print)
            result = await agent.run(user_prompt)
            print(result)
            result = await agent.run("Which tools does it have?")
            print(result)

    asyncio.run(main())
