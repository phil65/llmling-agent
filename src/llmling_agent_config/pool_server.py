"""Pool server configuration."""

from __future__ import annotations

from typing import Literal, assert_never

from pydantic import ConfigDict, Field
from schemez import Schema


TransportType = Literal["stdio", "sse", "streamable-http"]


class MCPPoolServerConfig(Schema):
    """Configuration for pool-based MCP server."""

    enabled: bool = Field(default=False, title="Server enabled")
    """Whether this server is currently enabled."""

    # Resource exposure control
    serve_nodes: list[str] | bool = Field(
        default=True,
        title="Serve nodes",
        examples=[["node1", "node2"], ["analysis", "transform"]],
    )
    """Which nodes to expose as tools:
    - True: All nodes
    - False: No nodes
    - list[str]: Specific node names
    """

    serve_prompts: list[str] | bool = Field(
        default=True,
        title="Serve prompts",
        examples=[["prompt1", "prompt2"], ["system", "user"]],
    )
    """Which prompts to expose:
    - True: All prompts from manifest
    - False: No prompts
    - list[str]: Specific prompt names
    """

    transport: TransportType = Field(
        default="stdio",
        title="Transport type",
        examples=["stdio", "sse", "streamable-http"],
    )
    """Transport type to use."""

    host: str = Field(
        default="localhost",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind server to (SSE / Streamable-HTTP only)."""

    port: int = Field(
        default=3001,
        gt=0,
        title="Server port",
        examples=[3001, 8080, 9000],
    )
    """Port to listen on (SSE / Streamable-HTTP only)."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        title="CORS origins",
        examples=[["*"], ["https://example.com", "https://app.com"]],
    )
    """Allowed CORS origins (SSE / Streamable-HTTP only)."""

    zed_mode: bool = Field(default=False, title="Zed editor mode")
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
            case _ as unreachable:
                assert_never(unreachable)

    def should_serve_prompt(self, name: str) -> bool:
        """Check if a prompt should be exposed."""
        match self.serve_prompts:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_prompts
            case _ as unreachable:
                assert_never(unreachable)
