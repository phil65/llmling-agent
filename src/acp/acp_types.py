"""ACP (Agent Client Protocol) types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acp.schema import McpServer, SessionUpdate  # noqa: TC001


@dataclass
class SessionData:
    """Complete session data for persistence and restoration."""

    session_id: str
    """Unique session identifier"""

    model_name: str
    """Currently chosen model"""

    session_mode: str | None
    """Currently selected session mode"""

    cwd: str
    """Working directory for the session"""

    mcp_servers: list[McpServer]
    """MCP server configurations"""

    notifications: list[SessionUpdate]
    """Complete notification history"""

    metadata: dict[str, Any]
    """Additional session metadata"""
