"""High-level functional interfaces for AgentPool."""

from agentpool.functional.run import (
    run_agent,
    run_agent_sync,
)
from agentpool.functional.structure import (
    get_structured,
    get_structured_multiple,
    pick_one,
)

__all__ = [
    "auto_callable",
    "get_structured",
    "get_structured_multiple",
    "pick_one",
    "run_agent",
    "run_agent_sync",
]
