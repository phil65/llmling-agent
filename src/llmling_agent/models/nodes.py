"""Team configuration models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from llmling_agent.models.events import DEFAULT_TEMPLATE, EventConfig
from llmling_agent.models.forward_targets import ForwardingTarget  # noqa: TC001
from llmling_agent.models.mcp_server import (
    MCPServerBase,
    MCPServerConfig,
    StdioMCPServer,
)


if TYPE_CHECKING:
    from llmling_agent.messaging.eventnode import Event


class NodeConfig(BaseModel):
    """Configuration for a Node of the messaging system."""

    name: str | None = None
    """Name of the Agent / Team"""

    description: str | None = None
    """Optional description of the agent / team."""

    triggers: list[EventConfig] = Field(default_factory=list)
    """Event sources that activate this agent / team"""

    connections: list[ForwardingTarget] = Field(default_factory=list)
    """Targets to forward results to."""

    mcp_servers: list[str | MCPServerConfig] = Field(default_factory=list)
    """List of MCP server configurations:
    - str entries are converted to StdioMCPServer
    - MCPServerConfig for full server configuration
    """

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Get processed MCP server configurations.

        Converts string entries to StdioMCPServer configs by splitting
        into command and arguments.

        Returns:
            List of MCPServerConfig instances

        Raises:
            ValueError: If string entry is empty
        """
        configs: list[MCPServerConfig] = []

        for server in self.mcp_servers:
            match server:
                case str():
                    parts = server.split()
                    if not parts:
                        msg = "Empty MCP server command"
                        raise ValueError(msg)

                    configs.append(StdioMCPServer(command=parts[0], args=parts[1:]))
                case MCPServerBase():
                    configs.append(server)

        return configs


class EventNodeConfig(NodeConfig):
    """Base configuration for event nodes.

    All event node configurations must:
    1. Specify their type for discrimination
    2. Implement get_event() to create their event instance
    """

    type: str = Field("event", init=False)
    """Discriminator field for event configs."""

    enabled: bool = True
    """Whether this event source is active."""

    template: str = DEFAULT_TEMPLATE
    """Jinja2 template for formatting events."""

    include_metadata: bool = True
    """Control metadata visibility in template."""

    include_timestamp: bool = True
    """Control timestamp visibility in template."""

    @abstractmethod
    def get_event(self) -> Event[Any]:
        """Create event instance from this configuration.

        This method should:
        1. Create the event instance
        2. Wrap any functions if needed
        3. Return the configured event
        """
        raise NotImplementedError
