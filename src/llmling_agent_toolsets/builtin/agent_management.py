"""Provider for agent management tools."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRetry,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.agent.events import ToolCallProgressEvent
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.tools.exceptions import ToolError
from llmling_agent.utils.result_utils import to_type


if TYPE_CHECKING:
    from llmling_agent.agent import Agent

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
    if not ctx.pool:
        msg = "Agent needs to be in a pool to delegate tasks"
        raise ToolError(msg)

    if agent_or_team_name in ctx.pool.nodes:
        node = ctx.pool.nodes[agent_or_team_name]
        result = await node.run(prompt)
        return result.format(style="detailed", show_costs=True)

    msg = (
        f"No agent or team found with name: {agent_or_team_name}. "
        f"Available agents: {', '.join(ctx.pool.nodes.keys())}"
    )
    raise ModelRetry(msg)


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

    if not ctx.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    model = model or ctx.agent.model_name
    agent = Agent[TDeps](name=name, model=model, system_prompt=system_prompt, agent_pool=ctx.pool)
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

    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    name = f"delegate_{uuid4().hex[:8]}"
    model = model or ctx.agent.model_name
    agent = Agent[TDeps](name=name, model=model, system_prompt=system_prompt, agent_pool=ctx.pool)

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
    output_type: str | None = None,
) -> str:
    """Add a new agent to the pool.

    Args:
        name: Name for the new agent
        system_prompt: System prompt defining agent's role/behavior
        model: Optional model override (uses default if not specified)
        tools: Imort paths of the tools the agent should have, if any.
        session: Session ID to recover conversation state from
        output_type: Name of response type from manifest (for structured output)

    Returns:
        Confirmation message about the created agent
    """
    assert ctx.pool, "No agent pool available"
    try:
        agent: Agent[Any, Any] = await ctx.pool.add_agent(
            name=name,
            system_prompt=system_prompt,
            model=model,
            tools=tools,
            output_type=to_type(output_type, responses=ctx.pool.manifest.responses),
            session=session,
        )
    except ValueError as e:  # for wrong tool imports
        raise ModelRetry(message=f"Error creating agent: {e}") from None
    return f"Created agent {agent.name} using model {agent.model_name}"


async def add_team(  # noqa: D417
    ctx: AgentContext,
    nodes: list[str],
    mode: Literal["sequential", "parallel"] = "sequential",
    name: str | None = None,
) -> str:
    """Create a team from existing agents.

    Args:
        nodes: Names of agents / sub-teams to include in team
        mode: How the team should operate:
            - sequential: Agents process in sequence (pipeline)
            - parallel: Agents process simultaneously
        name: Optional name for the team
    """
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    # Verify all agents exist
    for node_name in nodes:
        if node_name not in ctx.pool.nodes:
            msg = (
                f"No agent or team found with name: {node_name}. "
                f"Available nodes: {', '.join(ctx.pool.nodes.keys())}"
            )
            raise ModelRetry(msg)
    if mode == "sequential":
        ctx.pool.create_team_run(nodes, name=name)
    else:
        ctx.pool.create_team(nodes, name=name)
    mode_str = "pipeline" if mode == "sequential" else "parallel"
    return f"Created {mode_str} team with nodes: {', '.join(nodes)}"


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
    assert ctx.pool, "No agent pool available"
    if agent_name not in ctx.pool.nodes:
        msg = (
            f"No agent or team found with name: {agent_name}. "
            f"Available nodes: {', '.join(ctx.pool.nodes.keys())}"
        )
        raise ModelRetry(msg)
    agent = ctx.pool.get_agent(agent_name)
    aggregated: list[str] = []
    tool_call_id = ctx.tool_call_id or ""

    try:
        async for event in agent.run_stream(message, model=model, store_history=store_history):
            match event:
                case (
                    PartStartEvent(part=TextPart(content=delta))
                    | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
                ):
                    aggregated.append(delta)
                    progress = ToolCallProgressEvent(
                        tool_call_id=tool_call_id,
                        status="in_progress",
                        message="".join(aggregated),
                        tool_name=ctx.tool_name,
                    )
                    await ctx.events.emit_event(progress)

                case (
                    PartStartEvent(part=ThinkingPart(content=delta))
                    | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
                ):
                    if delta:
                        aggregated.append(f"💭 {delta}")
                        progress = ToolCallProgressEvent(
                            tool_call_id=tool_call_id,
                            status="in_progress",
                            message="".join(aggregated),
                            tool_name=ctx.tool_name,
                        )
                        await ctx.events.emit_event(progress)

                case FunctionToolCallEvent(part=part):
                    aggregated.append(f"\n🔧 Using tool: {part.tool_name}\n")
                    progress = ToolCallProgressEvent(
                        tool_call_id=tool_call_id,
                        status="in_progress",
                        message="".join(aggregated),
                        tool_name=ctx.tool_name,
                    )
                    await ctx.events.emit_event(progress)

                case FunctionToolResultEvent(
                    result=ToolReturnPart(content=content, tool_name=tool_name),
                ):
                    aggregated.append(f"✅ {tool_name}: {content}\n")
                    progress = ToolCallProgressEvent(
                        tool_call_id=tool_call_id,
                        status="in_progress",
                        message="".join(aggregated),
                        tool_name=ctx.tool_name,
                    )
                    await ctx.events.emit_event(progress)

                case FunctionToolResultEvent(
                    result=RetryPromptPart(tool_name=tool_name) as result,
                ):
                    error_message = result.model_response()
                    aggregated.append(f"❌ {tool_name or 'unknown'}: {error_message}\n")
                    progress = ToolCallProgressEvent(
                        tool_call_id=tool_call_id,
                        status="in_progress",
                        message="".join(aggregated),
                        tool_name=ctx.tool_name,
                    )
                    await ctx.events.emit_event(progress)

                case _:
                    pass

    except Exception as e:
        msg = f"Failed to ask agent {agent_name}: {e}"
        raise ModelRetry(msg) from e

    return "".join(aggregated).strip()


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

    Nodes can be agents or teams.

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
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    # Get the nodes
    if source not in ctx.pool.nodes:
        msg = (
            f"No agent or team found with name: {source}. "
            f"Available nodes: {', '.join(ctx.pool.nodes.keys())}"
        )
        raise ModelRetry(msg)
    if target not in ctx.pool.nodes:
        msg = (
            f"No agent or team found with name: {target}. "
            f"Available nodes: {', '.join(ctx.pool.nodes.keys())}"
        )
        raise ModelRetry(msg)

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


class AgentManagementTools(StaticResourceProvider):
    """Provider for agent management tools."""

    def __init__(self, name: str = "agent_management") -> None:
        super().__init__(name=name)
        for tool in [
            self.create_tool(delegate_to, category="other"),
            self.create_tool(list_available_agents, category="search"),
            self.create_tool(list_available_teams, category="search"),
            self.create_tool(create_worker_agent, category="other"),
            self.create_tool(spawn_delegate, category="other"),
            self.create_tool(add_agent, category="other"),
            self.create_tool(add_team, category="other"),
            self.create_tool(ask_agent, category="other"),
            self.create_tool(connect_nodes, category="other"),
        ]:
            self.add_tool(tool)


if __name__ == "__main__":
    # import logging
    from llmling_agent import AgentPool

    user_prompt = """Add a stdio MCP server:
// 	"command": "npx",
// 	"args": ["mcp-graphql"],
// 	"env": { "ENDPOINT": "https://diego.one/graphql" }

."""

    async def main() -> None:
        from llmling_agent_config.toolsets import IntegrationToolsetConfig

        async with AgentPool() as pool:
            toolsets = [IntegrationToolsetConfig()]
            toolset_providers = [config.get_provider() for config in toolsets]
            agent = await pool.add_agent(
                "X",
                toolsets=toolset_providers,
                model="openai:gpt-5-nano",
            )
            result = await agent.run(user_prompt)
            print(result)
            result = await agent.run("Which tools does it have?")
            print(result)

    asyncio.run(main())
