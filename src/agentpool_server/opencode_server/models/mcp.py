"""MCP and logging models."""

from typing import Any, Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class LogRequest(OpenCodeBaseModel):
    """Log entry request."""

    service: str
    level: Literal["debug", "info", "warn", "error"]
    message: str
    extra: dict[str, Any] | None = None


class MCPStatus(OpenCodeBaseModel):
    """MCP server status."""

    name: str
    status: Literal["connected", "disconnected", "error"]
    tools: list[str] = Field(default_factory=list)
    error: str | None = None


class McpResource(OpenCodeBaseModel):
    """MCP resource info matching OpenCode SDK McpResource type."""

    name: str
    """Name of the resource."""

    uri: str
    """URI identifying the resource location."""

    description: str | None = None
    """Optional description of the resource."""

    mime_type: str | None = None
    """MIME type of the resource content."""

    client: str
    """Name of the MCP client/server providing this resource."""
