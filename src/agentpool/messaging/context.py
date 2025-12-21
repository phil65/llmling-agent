"""Base class for message processing nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agentpool import AgentPool
    from agentpool.agents.base_agent import BaseAgent
    from agentpool.messaging import MessageNode
    from agentpool.models.manifest import AgentsManifest
    from agentpool.prompts.manager import PromptManager
    from agentpool.ui.base import InputProvider
    from agentpool_config.nodes import NodeConfig


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
    def agent(self) -> BaseAgent[TDeps, Any]:
        """Return agent node, type-narrowed to BaseAgent."""
        from agentpool.agents.base_agent import BaseAgent

        assert isinstance(self.node, BaseAgent)
        return self.node

    def get_input_provider(self) -> InputProvider:
        from agentpool.ui.stdlib_provider import StdlibInputProvider

        if self.input_provider:
            return self.input_provider
        if self.pool and self.pool._input_provider:
            return self.pool._input_provider
        return StdlibInputProvider()

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from manifest."""
        return self.definition.prompt_manager
