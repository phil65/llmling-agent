"""Runtime context models for Agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.messaging.context import NodeContext
from llmling_agent.prompts.conversion_manager import ConversionManager


if TYPE_CHECKING:
    from llmling import RuntimeConfig
    from mcp import types

    from llmling_agent import AgentPool
    from llmling_agent.agent import AnyAgent
    from llmling_agent.models.agents import AgentConfig
    from llmling_agent.tools.base import Tool
    from llmling_agent_input.base import InputProvider


ConfirmationResult = Literal["allow", "skip", "abort_run", "abort_chain"]


@dataclass(kw_only=True)
class AgentContext[TDeps = Any](NodeContext[TDeps]):
    """Runtime context for agent execution.

    Generically typed with AgentContext[Type of Dependencies]
    """

    config: AgentConfig
    """Current agent's specific configuration."""

    model_settings: dict[str, Any] = field(default_factory=dict)
    """Model-specific settings."""

    data: TDeps | None = None
    """Custom context data."""

    runtime: RuntimeConfig | None = None
    """Reference to the runtime configuration."""

    @classmethod
    def create_default(
        cls,
        name: str,
        deps: TDeps | None = None,
        pool: AgentPool | None = None,
        input_provider: InputProvider | None = None,
    ) -> AgentContext[TDeps]:
        """Create a default agent context with minimal privileges.

        Args:
            name: Name of the agent

            deps: Optional dependencies for the agent
            pool: Optional pool the agent is part of
            input_provider: Optional input provider for the agent
        """
        from llmling_agent.models import AgentConfig, AgentsManifest

        defn = AgentsManifest()
        cfg = AgentConfig(name=name)
        return cls(
            input_provider=input_provider,
            node_name=name,
            definition=defn,
            config=cfg,
            data=deps,
            pool=pool,
        )

    @cached_property
    def converter(self) -> ConversionManager:
        """Get conversion manager from global config."""
        return ConversionManager(self.definition.conversion)

    # TODO: perhaps add agent directly to context?
    @property
    def agent(self) -> AnyAgent[TDeps, Any]:
        """Get the agent instance from the pool."""
        assert self.pool, "No agent pool available"
        assert self.node_name, "No agent name available"
        return self.pool.agents[self.node_name]

    @property
    def process_manager(self):
        """Get process manager from pool."""
        assert self.pool, "No agent pool available"
        return self.pool.process_manager

    async def handle_confirmation(
        self,
        tool: Tool,
        args: dict[str, Any],
    ) -> ConfirmationResult:
        """Handle tool execution confirmation.

        Returns True if:
        - No confirmation handler is set
        - Handler confirms the execution
        """
        provider = self.get_input_provider()
        mode = self.config.requires_tool_confirmation
        if (mode == "per_tool" and not tool.requires_confirmation) or mode == "never":
            return "allow"
        history = self.agent.conversation.get_history() if self.pool else []
        return await provider.get_tool_confirmation(self, tool, args, history)

    async def handle_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Handle elicitation request for additional information."""
        provider = self.get_input_provider()
        history = self.agent.conversation.get_history() if self.pool else []
        return await provider.get_elicitation(self, params, history)
