"""Helper functions for ACP agent."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    import structlog

    from acp.schema import AgentCapabilities
    from acp.schema.mcp import McpServer


def filter_servers_by_capabilities(
    servers: Sequence[McpServer],
    agent_capabilities: AgentCapabilities | None,
    *,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> list[McpServer]:
    """Filter MCP servers based on agent's supported transports.

    Args:
        servers: ACP-schema MCP servers to filter (not agentpool_config types)
        agent_capabilities: Agent's capabilities (None = no HTTP/SSE support)
        logger: Optional logger for warnings about unsupported servers

    Returns:
        Servers compatible with the agent's capabilities
    """
    from acp.schema.mcp import HttpMcpServer, SseMcpServer

    # Check what transports are supported
    supports_http = (
        agent_capabilities
        and agent_capabilities.mcp_capabilities
        and agent_capabilities.mcp_capabilities.http
    )
    supports_sse = (
        agent_capabilities
        and agent_capabilities.mcp_capabilities
        and agent_capabilities.mcp_capabilities.sse
    )

    supported_servers: list[McpServer] = []
    unsupported_servers: list[tuple[McpServer, str]] = []

    for server in servers:
        if isinstance(server, HttpMcpServer) and not supports_http:
            unsupported_servers.append((server, "HTTP"))
        elif isinstance(server, SseMcpServer) and not supports_sse:
            unsupported_servers.append((server, "SSE"))
        else:
            # Stdio servers or supported transport types
            supported_servers.append(server)

    # Log warning if some servers were filtered out
    if unsupported_servers and logger:
        transports = ", ".join(sorted({t for _, t in unsupported_servers}))
        server_names = ", ".join(s.name for s, _ in unsupported_servers)
        logger.warning(
            "Agent does not support some MCP transports, skipping servers",
            unsupported_transports=transports,
            skipped_servers=server_names,
            unsupported_count=len(unsupported_servers),
            supported_http=supports_http,
            supported_sse=supports_sse,
        )

    return supported_servers
