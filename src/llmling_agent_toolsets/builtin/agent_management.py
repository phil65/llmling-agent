"""Provider for agent pool building tools."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import ModelRetry

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.tools.exceptions import ToolError
from llmling_agent.utils.result_utils import to_type


if TYPE_CHECKING:
    from llmling_agent.agents import Agent

logger = get_logger(__name__)


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
    tool_info = ctx.native_agent.register_worker(agent)
    return f"Created worker agent and registered as tool: {tool_info.name}"


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
    return f"Created agent **{agent.name}** using model **{agent.model_name}**"


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
    return f"Created **{mode_str}** team with nodes: **{', '.join(nodes)}**"


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
        f"Created connection from **{source}** to **{target}** "
        f"*(type={connection_type}, queued={queued}, "
        f"strategy={queue_strategy if queued else 'n/a'})*"
    )


class AgentManagementTools(StaticResourceProvider):
    """Provider for agent pool building tools."""

    def __init__(self, name: str = "agent_management") -> None:
        super().__init__(name=name)
        for tool in [
            self.create_tool(create_worker_agent, category="other", destructive=False),
            self.create_tool(add_agent, category="other", destructive=False),
            self.create_tool(add_team, category="other", destructive=False),
            self.create_tool(connect_nodes, category="other", destructive=False),
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
