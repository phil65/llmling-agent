"""Agent capabilities definition."""

from __future__ import annotations

from typing import Literal

from psygnal import EventedModel
from pydantic import ConfigDict


AccessLevel = Literal["none", "own", "all"]


class Capabilities(EventedModel):
    """Defines what operations an agent is allowed to perform."""

    # Agent / Team discovery and delegation
    can_list_agents: bool = False
    """Whether the agent can discover other available agents."""

    can_delegate_tasks: bool = False
    """Whether the agent can delegate tasks to other agents."""

    can_observe_agents: bool = False
    """Whether the agent can monitor other agents' activities."""

    can_list_teams: bool = False
    """Whether the agent can discover available teams."""

    # History and statistics access
    history_access: AccessLevel = "none"
    """Level of access to conversation history."""

    # Resource capabilities
    can_load_resources: bool = False
    """Whether the agent can load and access resource content."""

    can_list_resources: bool = False
    """Whether the agent can discover available resources."""

    can_read_files: bool = False
    """Whether the agent can read local and remote files."""

    can_list_directories: bool = False
    """Whether the agent can list directories and their contents."""

    # Tool management
    can_register_tools: bool = False
    """Whether the agent can register importable functions as tools."""

    can_register_code: bool = False
    """Whether the agent can create new tools from provided code."""

    can_install_packages: bool = False
    """Whether the agent can install Python packages for tools."""

    can_chain_tools: bool = False
    """Whether the agent can chain multiple tool calls into one."""

    # Execution

    can_execute_code: bool = False
    """Whether the agent can execute Python code (WARNING: No sandbox)."""

    can_execute_commands: bool = False
    """Whether the agent can execute CLI commands (use at your own risk)."""

    can_manage_processes: bool = False
    """Whether the agent can start and manage background processes."""

    # Agent creation
    can_create_workers: bool = False
    """Whether the agent can create worker agents (as tools)."""

    can_create_delegates: bool = False
    """Whether the agent can spawn temporary delegate agents."""

    can_add_agents: bool = False
    """Whether the agent can add aother agents to the pool."""

    can_ask_agents: bool = False
    """Whether the agent can ask other agents of the pool."""

    can_add_teams: bool = False
    """Whether the agent can add teams to the pool."""

    can_connect_nodes: bool = False
    """Whether the agent can add teams to the pool."""

    can_ask_user: bool = False
    """Whether the agent can ask the user clarifying questions during processing."""

    can_add_mcp_servers: bool = False
    """Whether the agent can add new MCP servers."""

    can_load_skills: bool = False
    """Whether the agent can discover and load Claude Code Skills."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")

    def __contains__(self, required: Capabilities) -> bool:
        """Check if these capabilities contain all required capabilities.

        Example:
            required in agent.capabilities  # Can agent fulfill requirements?
        """
        # Check all boolean capabilities using any() to find any missing required flag
        if any(
            getattr(required, field) and not getattr(self, field)
            for field in type(self).model_fields
            if isinstance(getattr(required, field), bool)
        ):
            return False

        # Check access levels (none < own < all)
        access_order = {"none": 0, "own": 1, "all": 2}
        required_level = access_order[required.history_access]
        self_level = access_order[self.history_access]
        return not required_level > self_level

    def has_capability(self, capability: str) -> bool:
        """Check if a specific capability is enabled.

        Args:
            capability: Name of capability to check.
                      Can be a boolean capability (e.g., "can_delegate_tasks")
                      or an access level (e.g., "history_access")
        """
        match capability:
            case str() if hasattr(self, capability):
                value = getattr(self, capability)
                return bool(value) if isinstance(value, bool) else value != "none"
            case _:
                msg = f"Unknown capability: {capability}"
                raise ValueError(msg)
