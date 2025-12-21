"""AG-UI server module for agentpool.

This module provides server implementation for exposing AgentPool agents
via the AG-UI protocol with each agent on its own route.
"""

from __future__ import annotations

from agentpool_server.agui_server.server import AGUIServer

__all__ = ["AGUIServer"]
