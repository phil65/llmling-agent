"""Base class for message processing nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.messaging import MessageNode
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import NodeConfig


@dataclass(kw_only=True)
class NodeContext[TDeps = object]:
    """Context for message processing nodes."""

    node_name: str
    """Name of the current node."""

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

    def get_input_provider(self) -> InputProvider:
        from llmling_agent.ui.stdlib_provider import StdlibInputProvider

        if self.input_provider:
            return self.input_provider
        if self.pool and self.pool._input_provider:
            return self.pool._input_provider
        return StdlibInputProvider()

    @property
    def node(self) -> MessageNode[TDeps, Any]:
        """Get the agent instance from the pool."""
        assert self.pool, "No agent pool available"
        assert self.node_name, "No agent name available"
        return self.pool[self.node_name]  # pyright: ignore

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from manifest."""
        return self.definition.prompt_manager
