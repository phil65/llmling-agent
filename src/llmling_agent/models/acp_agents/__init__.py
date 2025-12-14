"""ACP Agets."""

from typing import Annotated
from .non_mcp import RegularACPAgentConfigTypes
from .mcp_capable import MCPCapableACPAgentConfigTypes, MCPCapableACPAgentConfig
from .base import BaseACPAgentConfig, ACPAgentConfig
from pydantic import Field

# Union of all ACP agent config types
ACPAgentConfigTypes = Annotated[
    ACPAgentConfig | RegularACPAgentConfigTypes | MCPCapableACPAgentConfigTypes,
    Field(discriminator="type"),
]

__all__ = [
    "ACPAgentConfig",
    "ACPAgentConfigTypes",
    "BaseACPAgentConfig",
    "MCPCapableACPAgentConfig",
    "MCPCapableACPAgentConfigTypes",
    "RegularACPAgentConfigTypes",
]
