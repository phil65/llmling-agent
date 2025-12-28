"""Agent, command, MCP, LSP, formatter, and logging routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (  # noqa: TC001
    Agent,
    Command,
    LogRequest,
    MCPStatus,
)


router = APIRouter(tags=["agent"])


@router.get("/agent")
async def list_agents(state: StateDep) -> list[Agent]:
    """List available agents.

    TODO: Integrate with AgentPool.
    """
    _ = state  # unused for now
    return [
        Agent(
            id="default",
            name="Default Agent",
            description="Default AgentPool agent",
        )
    ]


@router.get("/command")
async def list_commands(state: StateDep) -> list[Command]:
    """List available slash commands.

    TODO: Integrate with AgentPool slash commands.
    """
    _ = state  # unused for now
    return []


@router.get("/mcp")
async def get_mcp_status(state: StateDep) -> dict[str, MCPStatus]:
    """Get MCP server status.

    TODO: Integrate with AgentPool MCP.
    """
    _ = state  # unused for now
    return {}


@router.post("/log")
async def log(request: LogRequest, state: StateDep) -> bool:
    """Write a log entry.

    TODO: Integrate with proper logging.
    """
    _ = state  # unused for now
    print(f"[{request.level}] {request.service}: {request.message}")
    return True


@router.get("/lsp")
async def get_lsp_status(state: StateDep) -> list[dict[str, Any]]:
    """Get LSP server status.

    Returns empty list - LSP not supported yet.
    """
    _ = state
    return []


@router.get("/formatter")
async def get_formatter_status(state: StateDep) -> list[dict[str, Any]]:
    """Get formatter status.

    Returns empty list - formatters not supported yet.
    """
    _ = state
    return []


@router.get("/provider/auth")
async def get_provider_auth(state: StateDep) -> dict[str, list[dict[str, Any]]]:
    """Get provider authentication methods.

    Returns empty dict - we handle auth differently.
    """
    _ = state
    return {}
