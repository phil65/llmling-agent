from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Capabilities(BaseModel):
    """Base capabilities configuration."""

    can_list_agents: bool = False
    """Whether the agent can discover other available agents."""

    can_delegate_tasks: bool = False
    """Whether the agent can delegate tasks to other agents."""

    can_observe_agents: bool = False
    """Whether the agent can monitor other agents' activities."""

    history_access: Literal["none", "own", "all"] = "own"
    """Level of access to conversation history.

    Levels:
    - none: No access to history
    - own: Can only access own conversations
    - all: Can access all agents' conversations
    """

    stats_access: Literal["none", "own", "all"] = "own"
    """Level of access to usage statistics.

    Levels:
    - none: No access to statistics
    - own: Can only view own statistics
    - all: Can view all agents' statistics
    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


BuiltinRole = Literal["overseer", "specialist", "assistant"]
"""Built-in role types with predefined capabilities.

Available roles:
- overseer: Full access to agent management and history
- specialist: Access to own history and statistics
- assistant: Basic access to own history only
"""

RoleName = BuiltinRole | str
"""Valid role names, either built-in or custom.

Can be either:
- A built-in role ("overseer", "specialist", "assistant")
- A custom role name defined in the configuration
"""
BUILTIN_ROLES: dict[BuiltinRole, Capabilities] = {
    "overseer": Capabilities(
        can_list_agents=True,
        can_delegate_tasks=True,
        can_observe_agents=True,
        history_access="all",
        stats_access="all",
    ),
    "specialist": Capabilities(
        history_access="own",
        stats_access="own",
    ),
    "assistant": Capabilities(
        history_access="own",
        stats_access="none",
    ),
}