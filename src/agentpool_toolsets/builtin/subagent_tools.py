"""Provider for subagent/task tools with streaming support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import ModelRetry

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.agents.events import StreamCompleteEvent, SubAgentEvent
from agentpool.agents.events.processors import batch_stream_deltas
from agentpool.log import get_logger
from agentpool.resource_providers import StaticResourceProvider
from agentpool.tools.exceptions import ToolError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agentpool.agents.events import RichAgentStreamEvent


logger = get_logger(__name__)


async def _stream_task(
    ctx: AgentContext,
    source_name: str,
    source_type: Literal["agent", "team_parallel", "team_sequential"],
    stream: AsyncIterator[RichAgentStreamEvent[Any]],
    *,
    batch_deltas: bool = False,
    depth: int = 1,
) -> str:
    """Stream a task's execution, emitting SubAgentEvents into parent stream.

    Args:
        ctx: Agent context for emitting events
        source_name: Name of the agent/team executing the task
        source_type: Whether source is "agent" or "team"
        stream: Async iterator of stream events from agent.run_stream()
        batch_deltas: If True, batch consecutive text/thinking deltas for fewer UI updates
        depth: Nesting depth for nested task delegation

    Returns:
        Final text content from the stream
    """
    if batch_deltas:
        stream = batch_stream_deltas(stream)

    final_content: str = ""
    async for event in stream:
        # Handle nested SubAgentEvents - increment depth
        if isinstance(event, SubAgentEvent):
            nested_event = SubAgentEvent(
                source_name=event.source_name,
                source_type=event.source_type,
                event=event.event,
                depth=event.depth + depth,
            )
            await ctx.events.emit_event(nested_event)
        else:
            # Wrap the event in SubAgentEvent
            subagent_event = SubAgentEvent(
                source_name=source_name,
                source_type=source_type,
                event=event,
                depth=depth,
            )
            await ctx.events.emit_event(subagent_event)

            # Extract final content from StreamCompleteEvent
            if isinstance(event, StreamCompleteEvent):
                content = event.message.content
                final_content = str(content) if content else ""

    return final_content


class SubagentTools(StaticResourceProvider):
    """Provider for task delegation tools with streaming progress."""

    def __init__(
        self,
        name: str = "subagent_tools",
        *,
        batch_stream_deltas: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._batch_stream_deltas = batch_stream_deltas
        for tool in [
            self.create_tool(
                self.list_available_nodes, category="search", read_only=True, idempotent=True
            ),
            self.create_tool(self.task, category="other"),
        ]:
            self.add_tool(tool)

    async def list_available_nodes(  # noqa: D417
        self,
        ctx: AgentContext,
        node_type: Literal["all", "agent", "team"] = "all",
        only_idle: bool = False,
    ) -> str:
        """List available agents and/or teams in the current pool.

        Args:
            node_type: Filter by node type - "all", "agent", or "team"
            only_idle: If True, only returns nodes that aren't currently busy

        Returns:
            List of node names that can be used with the task tool
        """
        if ctx.pool is None:
            msg = "No agent pool available"
            raise ToolError(msg)
        lines: list[str] = []
        if node_type in ("all", "agent"):
            agents = dict(ctx.pool.all_agents)
            if only_idle:
                agents = {n: a for n, a in agents.items() if not a.is_busy()}
            for name, agent in agents.items():
                lines.extend([
                    f"name: {name}",
                    "type: agent",
                    f"description: {agent.description or 'No description'}",
                    "---",
                ])

        if node_type in ("all", "team"):  # List teams
            teams = ctx.pool.teams
            if only_idle:
                teams = {name: team for name, team in teams.items() if not team.is_running}
            for name, team in teams.items():
                lines.extend([
                    f"name: {name}",
                    f"description: {team.description or 'No description'}",
                    "---",
                ])

        return "\n".join(lines) if lines else "No nodes available"

    async def task(  # noqa: D417
        self,
        ctx: AgentContext,
        agent_or_team: str,
        prompt: str,
        description: str,
    ) -> str:
        """Execute a task on an agent or team.

        Launch a task to be executed by a specialized agent or team. The task runs
        synchronously, streaming progress events, and returns the result when complete.

        Args:
            agent_or_team: The agent or team to execute the task
            prompt: The task instructions for the agent or team
            description: A short (3-5 words) description of the task

        Returns:
            The result of the task execution
        """
        from agentpool import Team, TeamRun
        from agentpool.agents.base_agent import BaseAgent
        from agentpool.common_types import SupportsRunStream

        _ = description  # Used for logging/tracking in future

        if ctx.pool is None:
            msg = "Agent needs to be in a pool to execute tasks"
            raise ToolError(msg)

        if agent_or_team not in ctx.pool.nodes:
            msg = (
                f"No agent or team found with name: {agent_or_team}. "
                f"Available nodes: {', '.join(ctx.pool.nodes.keys())}"
            )
            raise ModelRetry(msg)

        # Determine source type and get node
        node = ctx.pool.nodes[agent_or_team]
        match node:
            case Team():
                source_type: Literal["team_parallel", "team_sequential", "agent"] = "team_parallel"
            case TeamRun():
                source_type = "team_sequential"
            case BaseAgent():
                source_type = "agent"

        if not isinstance(node, SupportsRunStream):
            msg = f"Node {agent_or_team} does not support streaming"
            raise ToolError(msg)

        logger.info("Executing task", agent_or_team=agent_or_team, description=description)
        # Stream with SubAgentEvent wrapping
        return await _stream_task(
            ctx,
            source_name=agent_or_team,
            source_type=source_type,
            stream=node.run_stream(prompt),
            batch_deltas=self._batch_stream_deltas,
        )
