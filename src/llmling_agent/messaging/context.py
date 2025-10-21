"""Base class for message processing nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.messaging.messageemitter import MessageEmitter
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.storage import StorageManager
    from llmling_agent_config.nodes import NodeConfig
    from llmling_agent_input.base import InputProvider


ProgressCallback = Callable[[float, float | None, str], Awaitable[None]]


@dataclass(kw_only=True)
class NodeContext[TDeps]:
    """Context for message processing nodes."""

    node_name: str
    """Name of the current node."""

    pool: AgentPool[Any] | None = None
    """The agent pool the node is part of."""

    config: NodeConfig
    """Node configuration."""

    definition: AgentsManifest
    """Complete agent definition with all configurations."""

    current_prompt: str | None = None
    """Current prompt text for the agent."""

    # in_async_context: bool = False
    # """Whether we're running in an async context."""

    input_provider: InputProvider | None = None
    """Provider for human-input-handling."""

    def get_input_provider(self) -> InputProvider:
        from llmling_agent_input.stdlib_provider import StdlibInputProvider

        if self.input_provider:
            return self.input_provider
        if self.pool and self.pool._input_provider:
            return self.pool._input_provider
        return StdlibInputProvider()

    @property
    def node(self) -> MessageEmitter[TDeps, Any]:
        """Get the agent instance from the pool."""
        assert self.pool, "No agent pool available"
        assert self.node_name, "No agent name available"
        return self.pool[self.node_name]  # pyright: ignore

    @cached_property
    def storage(self) -> StorageManager:
        """Storage manager from pool or config."""
        from llmling_agent.storage import StorageManager

        if self.pool:
            return self.pool.storage
        return StorageManager(self.definition.storage)

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from manifest."""
        return self.definition.prompt_manager

    @property
    def report_progress(self) -> ProgressCallback | None:
        """Access progress reporting from pool server if available."""
        return (  # pyright: ignore[reportReturnType]
            self.pool.progress_handlers
            if self.pool and self.pool.progress_handlers
            else None
        )
