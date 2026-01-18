"""MCP server configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import ConfigDict, Field, HttpUrl, model_validator
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

    enabled_tools: list[str] | None = Field(
        default=None,
        examples=[["read_file", "list_directory"], ["search", "fetch"]],
        title="Enabled tools",
    )
    """If set, only these tools will be available (whitelist).
    Mutually exclusive with disabled_tools."""

    disabled_tools: list[str] | None = Field(
        default=None,
        examples=[["delete_file", "write_file"], ["dangerous_tool"]],
        title="Disabled tools",
    )
    """Tools to exclude from this server (blacklist). Mutually exclusive with enabled_tools."""

    @model_validator(mode="after")
    def _validate_tool_filters(self) -> Self:
        """Validate that enabled_tools and disabled_tools are mutually exclusive."""
        if self.enabled_tools is not None and self.disabled_tools is not None:
            raise ValueError("Cannot specify both 'enabled_tools' and 'disabled_tools'")
        return self

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed based on enabled/disabled lists.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed, False otherwise
        """
        if self.enabled_tools is not None:
            return tool_name in self.enabled_tools
        if self.disabled_tools is not None:
            return tool_name not in self.disabled_tools
        return True

    def needs_tool_filtering(self) -> bool:
        """Check if this config has tool filtering configured."""
        return self.enabled_tools is not None or self.disabled_tools is not None

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this MCP server with mcp-filter for tool filtering.

        Creates a new StdioMCPServerConfig that runs mcp-filter as a proxy,
        applying the configured enabled_tools/disabled_tools filtering.

        Returns:
            A new StdioMCPServerConfig that wraps the original server with mcp-filter

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for the server process."""
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def to_pydantic_ai(self) -> MCPServer:
        """Convert to pydantic-ai MCP server instance."""
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

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this stdio MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "stdio", "--stdio-command", self.command]

        # Add original args as a single --stdio-arg
        if self.args:
            filter_args.extend(["--stdio-arg", " ".join(self.args)])

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            env=self.env,
            timeout=self.timeout,
        )

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

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this SSE MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "http", "--http-url", str(self.url)]

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            timeout=self.timeout,
        )

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

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this HTTP MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "http", "--http-url", str(self.url)]

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            timeout=self.timeout,
        )

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


def parse_mcp_servers_json(data: dict[str, object]) -> list[MCPServerConfig]:
    """Parse MCP servers from JSON format used by clients (e.g., Zed).

    Expected format:
        {
            "mcpServers": {
                "server_name": {
                    "url": "https://...",
                    "transport": "sse" | "http"  # optional, defaults to http
                },
                ...
            }
        }

    Args:
        data: JSON data containing mcpServers key

    Returns:
        List of parsed MCPServerConfig instances

    Raises:
        ValueError: If data format is invalid or transport type unsupported
    """
    if "mcpServers" not in data:
        msg = "MCP config must contain 'mcpServers' key"
        raise ValueError(msg)

    servers: list[MCPServerConfig] = []
    mcp_servers = data["mcpServers"]
    if not isinstance(mcp_servers, dict):
        msg = "'mcpServers' must be an object"
        raise TypeError(msg)

    for server_name, server_cfg in mcp_servers.items():
        if not isinstance(server_cfg, dict):
            msg = f"Server config for '{server_name}' must be an object"
            raise TypeError(msg)

        url = server_cfg.get("url")
        if not url:
            msg = f"Server '{server_name}' must have a 'url' field"
            raise ValueError(msg)

        match server_cfg.get("transport"):
            case "sse":
                server: MCPServerConfig = SSEMCPServerConfig(name=server_name, url=url)
            case "http" | None:  # Default to HTTP
                server = StreamableHTTPMCPServerConfig(name=server_name, url=url)
            case unknown:
                msg = f"Unsupported transport type for '{server_name}': {unknown}"
                raise ValueError(msg)

        servers.append(server)

    return servers
