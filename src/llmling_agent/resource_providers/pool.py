"""Resource provider exposing Nodes as tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.prompts.prompts import BasePrompt
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo

logger = get_logger(__name__)


class PoolResourceProvider(ResourceProvider):
    """Provider that exposes an AgentPool's resources."""

    def __init__(
        self,
        pool: AgentPool[Any],
        name: str | None = None,
        zed_mode: bool = False,
        include_team_members: bool = False,
    ) -> None:
        """Initialize provider with agent pool.

        Args:
            pool: Agent pool to expose resources from
            name: Optional name override (defaults to pool name)
            zed_mode: Whether to enable Zed mode
            include_team_members: Whether to include team members in the pool
                                  in addition to the team itself.
        """
        super().__init__(name=name or repr(pool))
        self.pool = pool
        self.zed_mode = zed_mode
        self.include_team_members = include_team_members

    async def get_tools(self) -> list[Tool]:
        """Get tools from all agents in pool."""
        team_tools = [team.to_tool() for team in self.pool.teams.values()]
        agents = list(self.pool.agents.values())
        team_members = {member for t in self.pool.teams.values() for member in t.nodes}
        if self.include_team_members:
            agent_tools = [agent.to_tool() for agent in agents]
        else:
            agent_tools = [agent.to_tool() for agent in agents if agent not in team_members]
        return team_tools + agent_tools

    async def get_prompts(self) -> list[BasePrompt]:
        """Get prompts from pool's manifest."""
        prompts: list[Any] = []
        # if self.pool.manifest.prompts:
        #     prompts.extend(self.pool.manifest.prompts.system_prompts.values())

        # if self.zed_mode:
        #     prompts = prepare_prompts_for_zed(prompts)

        return prompts

    async def get_resources(self) -> list[ResourceInfo]:
        """Get resources from pool's manifest."""
        # Here we could expose knowledge bases or other resources from manifest
        return []
