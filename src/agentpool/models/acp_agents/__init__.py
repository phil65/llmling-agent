"""ACP Agets."""

from typing import Annotated
from .non_mcp import RegularACPAgentConfigTypes
from .mcp_capable import MCPCapableACPAgentConfigTypes
from .base import BaseACPAgentConfig, ACPAgentConfig
from pydantic import Field

# Union of all ACP agent config types (discriminated by 'provider')
ACPAgentConfigTypes = Annotated[
    ACPAgentConfig | RegularACPAgentConfigTypes | MCPCapableACPAgentConfigTypes,
    Field(discriminator="provider"),
]

__all__ = [
    "ACPAgentConfig",
    "ACPAgentConfigTypes",
    "BaseACPAgentConfig",
    "MCPCapableACPAgentConfigTypes",
    "RegularACPAgentConfigTypes",
]
