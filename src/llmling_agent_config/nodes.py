"""Team configuration models."""

from __future__ import annotations

from evented.configs import EventConfig
from pydantic import ConfigDict, Field, ImportString
from schemez import Schema

from llmling_agent.ui.base import InputProvider
from llmling_agent_config.forward_targets import ForwardingTarget
from llmling_agent_config.mcp_server import (
    BaseMCPServerConfig,
    MCPServerConfig,
    StdioMCPServerConfig,
)


class NodeConfig(Schema):
    """Configuration for a Node of the messaging system."""

    name: str | None = Field(
        default=None,
        examples=["main_agent", "web_searcher", "code_assistant"],
        title="Node name",
    )
    """Name of the Agent / Team"""

    # display_name: str | None = None
    # """Display Name of the Agent / Team"""

    description: str | None = Field(
        default=None,
        examples=["Main conversation agent", "Handles web search requests"],
        title="Node description",
    )
    """Optional description of the agent / team."""

    triggers: list[EventConfig] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "type": "time",
                    "name": "daily_check",
                    "schedule": "0 9 * * *",
                    "prompt": "Daily status update",
                }
            ],
            [
                {
                    "type": "file",
                    "name": "code_watcher",
                    "paths": ["./src"],
                    "extensions": [".py"],
                }
            ],
        ],
        title="Event triggers",
    )
    """Event sources that activate this agent / team"""

    connections: list[ForwardingTarget] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "type": "node",
                    "name": "output_agent",
                    "connection_type": "run",
                    "wait_for_completion": True,
                }
            ],
            [
                {
                    "type": "file",
                    "path": "logs/messages.txt",
                    "template": "{{ message.content }}",
                }
            ],
        ],
        title="Message forwarding targets",
    )
    """Targets to forward results to."""

    mcp_servers: list[str | MCPServerConfig] = Field(
        default_factory=list, title="MCP servers", examples=[["uvx some-server"]]
    )
    """List of MCP server configurations:
    - str entries are converted to StdioMCPServerConfig
    - MCPServerConfig for full server configuration
    """

    input_provider: ImportString[InputProvider] | None = Field(
        default=None,
        title="Input provider",
    )
    """Provider for human-input-handling."""

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Get processed MCP server configurations.

        Converts string entries to StdioMCPServerConfigs by splitting
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

                    configs.append(StdioMCPServerConfig(command=parts[0], args=parts[1:]))
                case BaseMCPServerConfig():
                    configs.append(server)

        return configs
