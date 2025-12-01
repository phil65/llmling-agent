"""Provider for subagent interaction tools with streaming support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


if TYPE_CHECKING:
    from llmling_agent.agent.events import RichAgentStreamEvent


logger = get_logger(__name__)


async def _stream_agent_with_progress(
    ctx: AgentContext,
    stream,
) -> str:
    """Stream an agent's execution and emit progress events.

    Args:
        ctx: Agent context for emitting events
        stream: Async iterator of stream events from agent.run_stream()

    Returns:
        Aggregated content from the stream
    """
    aggregated: list[str] = []
    tool_call_id = ctx.tool_call_id or ""

    async for event in stream:
        event: RichAgentStreamEvent
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

    return "".join(aggregated).strip()


async def list_available_nodes(  # noqa: D417
    ctx: AgentContext,
    node_type: Literal["all", "agent", "team"] = "all",
    only_idle: bool = False,
    detailed: bool = False,
) -> str:
    """List available agents and/or teams in the current pool.

    Args:
        node_type: Filter by node type - "all", "agent", or "team"
        only_idle: If True, only returns nodes that aren't currently busy
        detailed: If True, additional info for each node is provided (e.g. description)

    Returns:
        List of node names that you can use with delegate_to or ask_agent
    """
    from llmling_agent import TeamRun

    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    lines: list[str] = []

    # List agents
    if node_type in ("all", "agent"):
        agents = dict(ctx.pool.agents)
        if only_idle:
            agents = {name: agent for name, agent in agents.items() if not agent.is_busy()}
        if not detailed:
            lines.extend(agents.keys())
        else:
            for name, agent in agents.items():
                lines.extend([
                    f"name: {name}",
                    "type: agent",
                    f"description: {agent.description or 'No description'}",
                    f"model: {agent.model_name}",
                    "---",
                ])

    # List teams
    if node_type in ("all", "team"):
        teams = ctx.pool.teams
        if only_idle:
            teams = {name: team for name, team in teams.items() if not team.is_running}
        if not detailed:
            lines.extend(teams.keys())
        else:
            for name, team in teams.items():
                lines.extend([
                    f"name: {name}",
                    f"type: {'sequential' if isinstance(team, TeamRun) else 'parallel'} team",
                    f"description: {team.description or 'No description'}",
                    f"members: {', '.join(a.name for a in team.agents)}",
                    "---",
                ])

    return "\n".join(lines) if lines else "No nodes available"


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

    if agent_or_team_name not in ctx.pool.nodes:
        msg = (
            f"No agent or team found with name: {agent_or_team_name}. "
            f"Available nodes: {', '.join(ctx.pool.nodes.keys())}"
        )
        raise ModelRetry(msg)

    # For teams, use simple run() - no streaming support yet
    if agent_or_team_name in ctx.pool.teams:
        result = await ctx.pool.teams[agent_or_team_name].run(prompt)
        return result.format(style="detailed", show_costs=True)

    # For agents, stream with progress events
    agent = ctx.pool.agents[agent_or_team_name]
    return await _stream_agent_with_progress(ctx, agent.run_stream(prompt))


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
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    if agent_name not in ctx.pool.agents:
        msg = (
            f"Agent not found: {agent_name}. Available agents: {', '.join(ctx.pool.agents.keys())}"
        )
        raise ModelRetry(msg)

    agent = ctx.pool.get_agent(agent_name)

    try:
        stream = agent.run_stream(message, model=model, store_history=store_history)
        return await _stream_agent_with_progress(ctx, stream)
    except Exception as e:
        msg = f"Failed to ask agent {agent_name}: {e}"
        raise ModelRetry(msg) from e


class SubagentTools(StaticResourceProvider):
    """Provider for subagent interaction tools with streaming progress."""

    def __init__(self, name: str = "subagent_tools") -> None:
        super().__init__(name=name)
        for tool in [
            self.create_tool(list_available_nodes, category="search"),
            self.create_tool(delegate_to, category="other"),
            self.create_tool(ask_agent, category="other"),
        ]:
            self.add_tool(tool)
