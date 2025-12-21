"""MCP server configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import ConfigDict, Field, HttpUrl
from schemez import Schema


if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP


class MCPServerAuthSettings(Schema):
    """Represents authentication configuration for a server.

    Minimal OAuth v2.1 support with sensible defaults.
    """

    oauth: bool = Field(default=False, title="Enable OAuth")

    # Local callback server configuration
    redirect_port: int = Field(
        default=3030, ge=1, lt=65536, examples=[3030, 8080, 9000], title="Redirect port"
    )
    redirect_path: str = Field(
        default="/callback",
        examples=["/callback", "/auth/callback", "/oauth"],
        title="Redirect path",
    )

    # Optional scope override. If set to a list, values are space-joined.
    scope: str | list[str] | None = Field(
        default=None,
        examples=["read write", ["read", "write"], "admin"],
        title="OAuth scope",
    )

    # Token persistence: use OS keychain via 'keyring' by default; fallback to 'memory'.
    persist: Literal["keyring", "memory"] = Field(
        default="keyring",
        examples=["keyring", "memory"],
        title="Token persistence",
    )


class BaseMCPServerConfig(Schema):
    """Base model for MCP server configuration."""

    type: str = Field(title="Server type")
    """Type discriminator for MCP server configurations."""

    name: str | None = Field(
        default=None,
        examples=["my_server", "api_connector", "file_handler"],
        title="Server name",
    )
    """Optional name for referencing the server."""

    enabled: bool = Field(default=True, title="Server enabled")
    """Whether this server is currently enabled."""

    env: dict[str, str] | None = Field(default=None, title="Environment variables")
    """Environment variables to pass to the server process."""

    timeout: float = Field(
        default=60.0,
        gt=0,
        examples=[30.0, 60.0, 120.0],
        title="Server timeout",
    )
    """Timeout for the server process in seconds."""

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for the server process."""
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def to_pydantic_ai(self) -> MCPServer:
        """Convert to pydantic-ai MCP server instance.

        Returns:
            A pydantic-ai MCP server instance
        """
        raise NotImplementedError

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this server configuration."""
        raise NotImplementedError

    @classmethod
    def from_string(cls, text: str) -> MCPServerConfig:
        """Create a MCPServerConfig from a string."""
        text = text.strip()
        if text.startswith(("http://", "https://")) and text.endswith("/sse"):
            return SSEMCPServerConfig(url=HttpUrl(text))
        if text.startswith(("http://", "https://")):
            return StreamableHTTPMCPServerConfig(url=HttpUrl(text))
        return StdioMCPServerConfig.from_string(text)


class StdioMCPServerConfig(BaseMCPServerConfig):
    """MCP server started via stdio.

    Uses subprocess communication through standard input/output streams.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Stdio MCP Server"})

    type: Literal["stdio"] = Field("stdio", init=False)
    """Stdio server coniguration."""

    command: str = Field(
        examples=["python", "node", "pipx", "uvx"],
        title="Command to execute",
    )
    """Command to execute (e.g. "pipx", "python", "node")."""

    args: list[str] = Field(
        default_factory=list,
        examples=[["run", "mcp-server"], ["-m", "my_mcp_server"], ["--debug"]],
        title="Command arguments",
    )
    """Command arguments (e.g. ["run", "some-server", "--debug"])."""

    @classmethod
    def from_string(cls, command: str) -> Self:
        """Create a MCP server from a command string."""
        parts = command.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1].split() if len(parts) > 1 else []
        return cls(command=cmd, args=args)

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this stdio server configuration."""
        return f"{self.command}_{' '.join(self.args)}"

    def to_pydantic_ai(self) -> MCPServerStdio:
        """Convert to pydantic-ai MCPServerStdio instance."""
        from pydantic_ai.mcp import MCPServerStdio

        return MCPServerStdio(
            command=self.command,
            args=self.args,
            env=self.get_env_vars() if self.env else None,
            id=self.name,
            timeout=self.timeout,
        )


class SSEMCPServerConfig(BaseMCPServerConfig):
    """MCP server using Server-Sent Events transport.

    Connects to a server over HTTP with SSE for real-time communication.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "SSE MCP Server"})

    type: Literal["sse"] = Field("sse", init=False)
    """SSE server configuration."""

    url: HttpUrl = Field(
        examples=["https://api.example.com/sse", "http://localhost:8080/events"],
        title="SSE endpoint URL",
    )
    """URL of the SSE server endpoint."""

    headers: dict[str, str] | None = Field(default=None, title="HTTP headers")
    """Headers to send with the SSE request."""

    auth: MCPServerAuthSettings = Field(
        default_factory=MCPServerAuthSettings,
        title="Authentication settings",
    )
    """OAuth settings for the SSE server."""

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this SSE server configuration."""
        return f"sse_{self.url}"

    def to_pydantic_ai(self) -> MCPServerSSE:
        """Convert to pydantic-ai MCPServerSSE instance."""
        from pydantic_ai.mcp import MCPServerSSE

        url = str(self.url)
        return MCPServerSSE(url=url, headers=self.headers, id=self.name, timeout=self.timeout)


class StreamableHTTPMCPServerConfig(BaseMCPServerConfig):
    """MCP server using StreamableHttp.

    Connects to a server over HTTP with streamable HTTP.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Streamable HTTP MCP Server"})

    type: Literal["streamable-http"] = Field("streamable-http", init=False)
    """HTTP server configuration."""

    url: HttpUrl = Field(
        examples=["https://api.example.com/mcp", "http://localhost:8080/stream"],
        title="HTTP endpoint URL",
    )
    """URL of the HTTP server endpoint."""

    headers: dict[str, str] | None = Field(default=None, title="HTTP headers")
    """Headers to send with the HTTP request."""

    auth: MCPServerAuthSettings = Field(
        default_factory=MCPServerAuthSettings,
        title="Authentication settings",
    )
    """OAuth settings for the HTTP server."""

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this streamable HTTP server configuration."""
        return f"streamable_http_{self.url}"

    def to_pydantic_ai(self) -> MCPServerStreamableHTTP:
        """Convert to pydantic-ai MCPServerStreamableHTTP instance."""
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        return MCPServerStreamableHTTP(
            url=str(self.url),
            headers=self.headers,
            id=self.name,
            timeout=self.timeout,
        )


MCPServerConfig = Annotated[
    StdioMCPServerConfig | SSEMCPServerConfig | StreamableHTTPMCPServerConfig,
    Field(discriminator="type"),
]
