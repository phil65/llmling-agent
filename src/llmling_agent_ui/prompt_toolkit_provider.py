"""Prompt toolkit provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.agent.agent import Agent
from llmling_agent_ui.base import UIProvider


if TYPE_CHECKING:
    from llmling.config.store import ConfigStore

    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.messaging.messagenode import MessageNode
    from llmling_agent.models.ui import PromptToolkitUIConfig


class PromptToolkitUIProvider(UIProvider):
    """Interactive CLI using prompt-toolkit."""

    def __init__(self, config: PromptToolkitUIConfig):
        self.config = config

    def run_pool(self, pool: AgentPool) -> None:
        msg = "PromptToolkitUI only supports node mode"
        raise NotImplementedError(msg)

    def run_node(self, node: MessageNode) -> None:
        """Run prompt-toolkit interface for single node."""
        import asyncio

        from llmling_agent_cli.chat_session.session import start_interactive_session

        assert isinstance(node, Agent), "PromptToolkitUI only supports agents"
        asyncio.run(start_interactive_session(node, stream=self.config.stream))

    def run(self, store: ConfigStore | None = None) -> None:
        msg = "PromptToolkitUI only supports node mode"
        raise NotImplementedError(msg)
