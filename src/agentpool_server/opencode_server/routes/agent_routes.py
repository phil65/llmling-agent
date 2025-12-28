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
            name="default",
            description="Default AgentPool agent",
            mode="primary",
            default=True,
        )
    ]


@router.get("/command")
async def list_commands(state: StateDep) -> list[Command]:
    """List available slash commands.

    Commands are derived from MCP prompts available to the agent.
    """
    if state.agent is None or not hasattr(state.agent, "tools"):
        return []

    try:
        prompts = await state.agent.tools.list_prompts()
        return [
            Command(
                name=prompt.name,
                description=prompt.description or None,
            )
            for prompt in prompts
        ]
    except Exception:  # noqa: BLE001
        return []


@router.get("/mcp")
async def get_mcp_status(state: StateDep) -> dict[str, MCPStatus]:
    """Get MCP server status.

    Returns status for each connected MCP server.
    """
    from agentpool.mcp_server.manager import MCPManager
    from agentpool.resource_providers import AggregatingResourceProvider
    from agentpool.resource_providers.mcp_provider import MCPResourceProvider

    if state.agent is None or not hasattr(state.agent, "tools"):
        return {}

    def add_mcp_status(provider: MCPResourceProvider, result: dict[str, MCPStatus]) -> None:
        """Add status for a single MCP provider."""
        status_dict = provider.get_status()
        status_type = status_dict.get("status", "disabled")
        if status_type == "connected":
            result[provider.name] = MCPStatus(name=provider.name, status="connected")
        elif status_type == "failed":
            result[provider.name] = MCPStatus(
                name=provider.name,
                status="error",
                error=status_dict.get("error", "Unknown error"),
            )
        else:
            result[provider.name] = MCPStatus(name=provider.name, status="disabled")

    result: dict[str, MCPStatus] = {}
    try:
        for provider in state.agent.tools.external_providers:
            if isinstance(provider, MCPResourceProvider):
                add_mcp_status(provider, result)
            elif isinstance(provider, AggregatingResourceProvider):
                # Check nested providers in aggregating provider
                for nested in provider.providers:
                    if isinstance(nested, MCPResourceProvider):
                        add_mcp_status(nested, result)
            elif isinstance(provider, MCPManager):
                # MCPManager wraps multiple MCPResourceProviders
                for mcp_provider in provider.get_mcp_providers():
                    add_mcp_status(mcp_provider, result)
    except Exception:  # noqa: BLE001
        pass

    return result


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
