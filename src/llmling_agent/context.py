from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llmling_agent.config.capabilities import Capabilities
from llmling_agent.models import AgentConfig, AgentDefinition


class AgentContext(BaseModel):
    """Runtime context for agent execution."""

    agent_name: str
    """Name of the current agent."""

    capabilities: Capabilities
    """Current agent's capabilities."""

    definition: AgentDefinition
    """Complete agent definition with all configurations."""

    config: AgentConfig
    """Current agent's specific configuration."""

    model_settings: dict[str, Any]
    """Model-specific settings."""

    def get_capabilities(self) -> Capabilities:
        """Get the current agent's capabilities."""
        return self.capabilities

    @classmethod
    def create_default(
        cls,
        name: str,
        capabilities: Capabilities | None = None,
    ) -> AgentContext:
        """Create a default agent context with minimal privileges.

        Args:
            name: Name of the agent
            capabilities: Optional custom capabilities (defaults to minimal access)
        """
        caps = capabilities or Capabilities(
            history_access="none",
            stats_access="none",
            can_list_agents=False,
            can_delegate_tasks=False,
            can_observe_agents=False,
        )
        return cls(
            agent_name=name,
            capabilities=caps,
            definition=AgentDefinition(responses={}, agents={}, roles={}),
            config=AgentConfig(name=name, role="assistant"),
            model_settings={},
        )