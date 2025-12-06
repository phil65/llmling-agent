"""Provider for worker agent tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.agent.context import AgentContext
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent.tools.exceptions import ToolError


if TYPE_CHECKING:
    from llmling_agent_config.workers import WorkerConfig

logger = get_logger(__name__)


class WorkersTools(ResourceProvider):
    """Provider for worker agent tools.

    Creates tools for each configured worker that delegate to agents/teams in the pool.
    Tools are created lazily when get_tools() is called, using AgentContext to access
    the pool at call time.
    """

    def __init__(
        self,
        workers: list[WorkerConfig],
        name: str = "workers",
    ) -> None:
        """Initialize workers toolset.

        Args:
            workers: List of worker configurations
            name: Provider name
        """
        super().__init__(name=name)
        self.workers = workers

    async def get_tools(self) -> list[Tool]:
        """Get tools for all configured workers."""
        tools: list[Tool] = []
        for worker_config in self.workers:
            tool = self._create_worker_tool(worker_config)
            tools.append(tool)
        return tools

    def _create_worker_tool(self, worker_config: WorkerConfig) -> Tool:
        """Create a tool for a single worker configuration."""
        from llmling_agent_config.workers import (
            ACPAgentWorkerConfig,
            AgentWorkerConfig,
            AGUIAgentWorkerConfig,
            TeamWorkerConfig,
        )

        worker_name = worker_config.name

        match worker_config:
            case TeamWorkerConfig():
                return self._create_team_tool(worker_name)
            case AgentWorkerConfig():
                return self._create_agent_tool(
                    worker_name,
                    reset_history_on_run=worker_config.reset_history_on_run,
                    pass_message_history=worker_config.pass_message_history,
                )
            case ACPAgentWorkerConfig():
                return self._create_acp_agent_tool(worker_name)
            case AGUIAgentWorkerConfig():
                return self._create_agui_agent_tool(worker_name)

    def _create_team_tool(self, team_name: str) -> Tool:
        """Create tool for a team worker."""

        async def call_team(ctx: AgentContext, prompt: str) -> str:
            """Delegate task to team worker."""
            if not ctx.pool:
                msg = "No agent pool available"
                raise ToolError(msg)
            if team_name not in ctx.pool.teams:
                msg = f"Team {team_name!r} not found in pool"
                raise ToolError(msg)

            team = ctx.pool.teams[team_name]
            result = await team.run(prompt)
            return result.format(style="detailed", show_costs=True)

        tool_name = f"ask_{team_name}"
        normalized_name = team_name.replace("_", " ").title()
        docstring = f"Delegate task to team: {normalized_name}"

        call_team.__name__ = tool_name
        call_team.__doc__ = docstring

        return Tool.from_callable(
            call_team,
            name_override=tool_name,
            description_override=docstring,
            source="worker",
        )

    def _create_agent_tool(
        self,
        agent_name: str,
        *,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
    ) -> Tool:
        """Create tool for an agent worker."""

        async def call_agent(ctx: AgentContext, prompt: str) -> Any:
            """Delegate task to agent worker."""
            if not ctx.pool:
                msg = "No agent pool available"
                raise ToolError(msg)
            if agent_name not in ctx.pool.agents:
                msg = f"Agent {agent_name!r} not found in pool"
                raise ToolError(msg)

            worker = ctx.pool.agents[agent_name]

            if pass_message_history:
                parent = ctx.agent
                old_history = worker.conversation.get_history()
                worker.conversation.set_history(parent.conversation.get_history())

            if reset_history_on_run and not pass_message_history:
                worker.conversation.clear()

            try:
                result = await worker.run(prompt)
                return result.data
            finally:
                if pass_message_history:
                    worker.conversation.set_history(old_history)

        tool_name = f"ask_{agent_name}"
        normalized_name = agent_name.replace("_", " ").title()
        docstring = f"Get expert answer from specialized agent: {normalized_name}"

        # Get description from pool if available (will be resolved at call time)
        call_agent.__name__ = tool_name
        call_agent.__doc__ = docstring

        return Tool.from_callable(
            call_agent,
            name_override=tool_name,
            description_override=docstring,
            source="worker",
        )

    def _create_acp_agent_tool(self, agent_name: str) -> Tool:
        """Create tool for an ACP agent worker."""

        async def call_acp_agent(ctx: AgentContext, prompt: str) -> str:
            """Delegate task to ACP agent worker."""
            if not ctx.pool:
                msg = "No agent pool available"
                raise ToolError(msg)
            if agent_name not in ctx.pool.acp_agents:
                msg = f"ACP agent {agent_name!r} not found in pool"
                raise ToolError(msg)

            agent = ctx.pool.acp_agents[agent_name]
            result = await agent.run(prompt)
            return str(result.data)

        tool_name = f"ask_{agent_name}"
        normalized_name = agent_name.replace("_", " ").title()
        docstring = f"Get answer from ACP agent: {normalized_name}"

        call_acp_agent.__name__ = tool_name
        call_acp_agent.__doc__ = docstring

        return Tool.from_callable(
            call_acp_agent,
            name_override=tool_name,
            description_override=docstring,
            source="worker",
        )

    def _create_agui_agent_tool(self, agent_name: str) -> Tool:
        """Create tool for an AG-UI agent worker."""

        async def call_agui_agent(ctx: AgentContext, prompt: str) -> str:
            """Delegate task to AG-UI agent worker."""
            if not ctx.pool:
                msg = "No agent pool available"
                raise ToolError(msg)
            if agent_name not in ctx.pool.agui_agents:
                msg = f"AG-UI agent {agent_name!r} not found in pool"
                raise ToolError(msg)

            agent = ctx.pool.agui_agents[agent_name]
            result = await agent.run(prompt)
            return str(result.data)

        tool_name = f"ask_{agent_name}"
        normalized_name = agent_name.replace("_", " ").title()
        docstring = f"Get answer from AG-UI agent: {normalized_name}"

        call_agui_agent.__name__ = tool_name
        call_agui_agent.__doc__ = docstring

        return Tool.from_callable(
            call_agui_agent,
            name_override=tool_name,
            description_override=docstring,
            source="worker",
        )
