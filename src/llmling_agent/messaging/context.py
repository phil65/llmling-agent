"""Base class for message processing nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llmling_agent import Agent, AgentPool
    from llmling_agent.agents.acp_agent import ACPAgent
    from llmling_agent.agents.agui_agent import AGUIAgent
    from llmling_agent.messaging import MessageNode
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import NodeConfig


@dataclass(kw_only=True)
class NodeContext[TDeps = object]:
    """Context for message processing nodes."""

    node: MessageNode[TDeps, Any]
    """Current Node."""

    pool: AgentPool[Any] | None = None
    """The agent pool the node is part of."""

    config: NodeConfig
    """Node configuration."""

    definition: AgentsManifest
    """Complete agent definition with all configurations."""

    input_provider: InputProvider | None = None
    """Provider for human-input-handling."""

    data: TDeps | None = None
    """Custom context data."""

    @property
    def node_name(self) -> str:
        """Name of the current node."""
        return self.node.name

    @property
    def any_agent(self) -> Agent[TDeps, Any] | ACPAgent | AGUIAgent:
        """Return any-agent node, type-narrowed."""
        from llmling_agent import Agent
        from llmling_agent.agents.acp_agent import ACPAgent
        from llmling_agent.agents.agui_agent import AGUIAgent

        assert isinstance(self.node, Agent | ACPAgent | AGUIAgent)
        return self.node

    def get_input_provider(self) -> InputProvider:
        from llmling_agent.ui.stdlib_provider import StdlibInputProvider

        if self.input_provider:
            return self.input_provider
        if self.pool and self.pool._input_provider:
            return self.pool._input_provider
        return StdlibInputProvider()

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from manifest."""
        return self.definition.prompt_manager
