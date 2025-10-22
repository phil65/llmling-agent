"""Pool server configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field
from schemez import Schema


TransportType = Literal["stdio", "sse", "streamable-http"]


class MCPPoolServerConfig(Schema):
    """Configuration for pool-based MCP server."""

    enabled: bool = False
    """Whether this server is currently enabled."""

    # Resource exposure control
    serve_nodes: list[str] | bool = True
    """Which nodes to expose as tools:
    - True: All nodes
    - False: No nodes
    - list[str]: Specific node names
    """

    serve_prompts: list[str] | bool = True
    """Which prompts to expose:
    - True: All prompts from manifest
    - False: No prompts
    - list[str]: Specific prompt names
    """

    transport: TransportType = "stdio"
    """Transport type to use."""

    host: str = "localhost"
    """Host to bind server to (SSE / Streamable-HTTP only)."""

    port: int = Field(default=3001, gt=0)
    """Port to listen on (SSE / Streamable-HTTP only)."""

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    """Allowed CORS origins (SSE / Streamable-HTTP only)."""

    zed_mode: bool = False
    """Enable Zed editor compatibility mode."""

    model_config = ConfigDict(frozen=True)

    def should_serve_node(self, name: str) -> bool:
        """Check if a node should be exposed."""
        match self.serve_nodes:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_nodes
            case _:
                return False

    def should_serve_prompt(self, name: str) -> bool:
        """Check if a prompt should be exposed."""
        match self.serve_prompts:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_prompts
            case _:
                return False
