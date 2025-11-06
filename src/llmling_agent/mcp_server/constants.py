"""MCP related constants."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import mcp


SERVER_NAME = "llmling-server"
SERVER_CMD = [sys.executable, "-m", "mcp_server_llmling", "start"]

MCP_TO_LOGGING: dict[mcp.LoggingLevel, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}


# Map Python logging levels to MCP logging levels
LOGGING_TO_MCP: dict[int, mcp.LoggingLevel] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}
