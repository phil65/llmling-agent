"""Base class for message processing nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.agents.base_agent import BaseAgent
    from llmling_agent.messaging import MessageNode
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.tools.base import Tool
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import NodeConfig


ConfirmationResult = Literal["allow", "skip", "abort_run", "abort_chain"]


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

    tool_name: str | None = None
    """Name of the currently executing tool."""

    tool_call_id: str | None = None
    """ID of the current tool call."""

    tool_input: dict[str, Any] = field(default_factory=dict)
    """Input arguments for the current tool call."""

    @property
    def node_name(self) -> str:
        """Name of the current node."""
        return self.node.name

    @property
    def agent(self) -> BaseAgent[TDeps, Any]:
        """Return agent node, type-narrowed to BaseAgent."""
        from llmling_agent.agents.base_agent import BaseAgent

        assert isinstance(self.node, BaseAgent)
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

    async def handle_confirmation(self, tool: Tool, args: dict[str, Any]) -> ConfirmationResult:
        """Handle tool execution confirmation.

        Returns "allow" if:
        - No confirmation handler is set
        - Handler confirms the execution

        Args:
            tool: The tool being executed
            args: Arguments passed to the tool

        Returns:
            Confirmation result indicating how to proceed
        """
        from llmling_agent.agents.base_agent import BaseAgent

        provider = self.get_input_provider()

        # Get confirmation mode from agent if available
        if isinstance(self.node, BaseAgent):
            mode = self.node.tool_confirmation_mode
            if (mode == "per_tool" and not tool.requires_confirmation) or mode == "never":
                return "allow"
            history = self.node.conversation.get_history() if self.pool else []
        else:
            # Non-agent nodes default to allowing
            history = []

        return await provider.get_tool_confirmation(self, tool, args, history)
