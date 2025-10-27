"""MCP schema definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from acp.base import AnnotatedObject, Schema
from acp.schema.common import EnvVariable


class HttpHeader(AnnotatedObject):
    """An HTTP header to set when making requests to the MCP server."""

    name: str
    """The name of the HTTP header."""

    value: str
    """The value to set for the HTTP header."""


class BaseMcpServer(Schema):
    """MCP server base class."""

    name: str
    """Human-readable name identifying this MCP server."""


class HttpMcpServer(BaseMcpServer):
    """HTTP transport configuration.

    Only available when the Agent capabilities indicate `mcp_capabilities.http` is `true`.
    """

    headers: Sequence[HttpHeader]
    """HTTP headers to set when making requests to the MCP server."""

    type: Literal["http"] = "http"

    url: str
    """URL to the MCP server."""


class SseMcpServer(BaseMcpServer):
    """SSE transport configuration.

    Only available when the Agent capabilities indicate `mcp_capabilities.sse` is `true`.
    """

    headers: Sequence[HttpHeader]
    """HTTP headers to set when making requests to the MCP server."""

    type: Literal["sse"] = "sse"

    url: str
    """URL to the MCP server."""


class StdioMcpServer(BaseMcpServer):
    """Stdio transport configuration.

    All Agents MUST support this transport.
    """

    args: Sequence[str]
    """Command-line arguments to pass to the MCP server."""

    command: str
    """Path to the MCP server executable."""

    env: Sequence[EnvVariable]
    """Environment variables to set when launching the MCP server."""


McpServer = HttpMcpServer | SseMcpServer | StdioMcpServer
