"""Agent, command, MCP, LSP, formatter, and logging routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
import httpx
from llmling_models.auth.anthropic_auth import (
    AnthropicTokenStore,
    build_authorization_url,
    exchange_code_for_token,
    generate_pkce,
)
from pydantic import BaseModel, HttpUrl

from agentpool.mcp_server.manager import MCPManager
from agentpool.resource_providers import AggregatingResourceProvider
from agentpool_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)
from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import (
    Agent,
    Command,
    LogRequest,
    McpResource,
    MCPStatus,
)


router = APIRouter(tags=["agent"])


@router.get("/agent")
async def list_agents(state: StateDep) -> list[Agent]:
    """List available agents from the AgentPool.

    Returns all agents with their configurations, suitable for the agent
    switcher UI. Agents are marked as primary (visible in switcher) or
    subagent (hidden, used internally).
    """
    pool = state.agent.agent_pool
    assert pool is not None, "AgentPool is not initialized"
    agents = [
        Agent(
            name=name,
            description=agent.description or f"Agent: {name}",
            # model=AgentModel(model_id=agent.model_name or "unknown", provider_id=""),
            mode="primary",  # All agents visible for now; add hidden config later
            default=(name == pool.main_agent.name),  # Default agent from pool
        )
        for name, agent in pool.all_agents.items()
    ]
    return (
        agents
        if agents
        else [Agent(name="default", description="Default agent", mode="primary", default=True)]
    )


@router.get("/command")
async def list_commands(state: StateDep) -> list[Command]:
    """List available slash commands.

    Commands are derived from MCP prompts available to the agent.
    """
    try:
        prompts = await state.agent.tools.list_prompts()
        return [Command(name=p.name, description=p.description or "") for p in prompts]
    except Exception:  # noqa: BLE001
        return []


@router.get("/mcp")
async def get_mcp_status(state: StateDep) -> dict[str, MCPStatus]:
    """Get MCP server status.

    Returns status for each connected MCP server.
    """
    # Use agent's get_mcp_server_info method which handles different agent types
    server_info = await state.agent.get_mcp_server_info()

    # Convert MCPServerStatus dataclass to MCPStatus response model
    return {
        name: MCPStatus(name=status.name, status=status.status, error=status.error)
        for name, status in server_info.items()
    }


class AddMCPServerRequest(BaseModel):
    """Request to add an MCP server dynamically."""

    command: str | None = None
    """Command to run (for stdio servers)."""

    args: list[str] | None = None
    """Arguments for the command."""

    url: str | None = None
    """URL for HTTP/SSE servers."""

    env: dict[str, str] | None = None
    """Environment variables for the server."""


@router.post("/mcp")
async def add_mcp_server(request: AddMCPServerRequest, state: StateDep) -> MCPStatus:
    """Add an MCP server dynamically.

    Supports stdio servers (command + args) or HTTP/SSE servers (url).
    """
    # Build the config based on request
    # Note: client_id is auto-generated from command/url, custom names not supported
    config: SSEMCPServerConfig | StdioMCPServerConfig | StreamableHTTPMCPServerConfig
    if request.url:
        # HTTP-based server
        if request.url.endswith("/sse"):
            config = SSEMCPServerConfig(url=HttpUrl(request.url))
        else:
            config = StreamableHTTPMCPServerConfig(url=HttpUrl(request.url))
    elif request.command:  # Stdio server
        args = request.args or []
        config = StdioMCPServerConfig(command=request.command, args=args, env=request.env)
    else:
        detail = "Must provide either 'command' (for stdio) or 'url' (for HTTP/SSE)"
        raise HTTPException(status_code=400, detail=detail)

    # Find the MCPManager and add the server
    manager: MCPManager | None = None
    for provider in state.agent.tools.external_providers:
        if isinstance(provider, MCPManager):
            manager = provider
            break
        if isinstance(provider, AggregatingResourceProvider):
            for nested in provider.providers:
                if isinstance(nested, MCPManager):
                    manager = nested
                    break

    if manager is None:
        raise HTTPException(status_code=400, detail="No MCP manager available")

    try:
        await manager.setup_server(config, add_to_config=True)
        return MCPStatus(name=config.client_id, status="connected")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add MCP server: {e}") from e


@router.post("/log")
async def log(request: LogRequest, state: StateDep) -> bool:
    """Write a log entry.

    TODO: Integrate with proper logging.
    """
    _ = state  # unused for now
    print(f"[{request.level}] {request.service}: {request.message}")
    return True


@router.get("/experimental/resource")
async def list_mcp_resources(state: StateDep) -> dict[str, McpResource]:
    """Get all available MCP resources from connected servers.

    Returns a dictionary mapping resource keys to McpResource objects.
    Keys are formatted as "{client}:{resource_name}" for uniqueness.
    """
    try:
        result: dict[str, McpResource] = {}
        for resource in await state.agent.tools.list_resources():
            # Create unique key: sanitize client and resource names
            client_name = (resource.client or "unknown").replace("/", "_")
            resource_name = resource.name.replace("/", "_")
            result[f"{client_name}:{resource_name}"] = McpResource(
                name=resource.name,
                uri=resource.uri,
                description=resource.description,
                mime_type=resource.mime_type,
                client=resource.client or "unknown",
            )
    except Exception:  # noqa: BLE001
        return {}
    else:
        return result


@router.get("/experimental/tool/ids")
async def list_tool_ids(state: StateDep) -> list[str]:
    """List all available tool IDs.

    Returns a list of tool names that are available to the agent.
    OpenCode expects: Array<string>
    """
    try:
        tools = await state.agent.tools.get_tools()
        return [tool.name for tool in tools]
    except Exception:  # noqa: BLE001
        return []


class ToolListItem(BaseModel):
    """Tool info matching OpenCode SDK ToolListItem type."""

    id: str
    description: str
    parameters: dict[str, Any]


@router.get("/experimental/tool")
async def list_tools_with_schemas(  # noqa: D417
    state: StateDep,
    provider: str | None = None,
    model: str | None = None,
) -> list[ToolListItem]:
    """List tools with their JSON schemas.

    Args:
        provider: Optional provider filter (not used currently)
        model: Optional model filter (not used currently)

    Returns list of tools matching OpenCode's ToolListItem format:
    - id: string
    - description: string
    - parameters: unknown (JSON schema)
    """
    _ = provider, model  # Currently unused, for future filtering

    try:
        result = []
        for tool in await state.agent.tools.get_tools():
            # Extract parameters schema from the OpenAI function schema
            schema = tool.schema
            params = schema.get("function", {}).get("parameters", {})
            item = ToolListItem(id=tool.name, description=tool.description or "", parameters=params)
            result.append(item)
    except Exception:  # noqa: BLE001
        return []
    else:
        return result


@router.get("/lsp")
async def get_lsp_status(state: StateDep) -> list[dict[str, Any]]:
    """Get LSP server status.

    Returns status of all running LSP servers.
    """
    servers = []
    for server_id, server_state in state.lsp_manager._servers.items():
        # OpenCode TUI expects "connected" or "error" for status colors
        status = "connected" if server_state.initialized else "error"
        servers.append({
            "id": server_id,
            "name": server_id,
            "status": status,
            "language": server_state.language,
            "root": server_state.root_uri,  # TUI uses "root" not "rootUri"
        })
    return servers


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

    Returns available OAuth providers with their auth methods.
    """
    _ = state
    return {
        "anthropic": [
            {
                "type": "oauth",
                "label": "Connect Claude Max/Pro",
                "method": "code",
            }
        ],
        "copilot": [
            {
                "type": "oauth",
                "label": "Connect GitHub Copilot",
                "method": "device_code",
            }
        ],
    }


# Store for active OAuth flows (in production, use Redis or similar)
_oauth_flows: dict[str, dict[str, Any]] = {}


@router.post("/provider/{provider_id}/oauth/authorize")
async def oauth_authorize(provider_id: str, state: StateDep) -> dict[str, Any]:
    """Start OAuth authorization flow for a provider.

    Returns URL and instructions for the user to complete authorization.
    """
    _ = state

    if provider_id == "anthropic":
        verifier, challenge = generate_pkce()
        auth_url = build_authorization_url(verifier, challenge)
        # Store verifier for callback
        _oauth_flows[f"anthropic:{verifier}"] = {"verifier": verifier}
        return {
            "url": auth_url,
            "instructions": "Sign in with your Anthropic account and copy the authorization code",
            "method": "code",
            "state": verifier,
        }

    if provider_id == "copilot":
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://github.com/login/device/code",
                headers={
                    "accept": "application/json",
                    "editor-version": "Neovim/0.6.1",
                    "editor-plugin-version": "copilot.vim/1.16.0",
                    "content-type": "application/json",
                    "user-agent": "GithubCopilot/1.155.0",
                },
                json={"client_id": "Iv1.b507a08c87ecfe98", "scope": "read:user"},
            )
            resp.raise_for_status()
            data = resp.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data["verification_uri"]
        # Store device_code for callback
        _oauth_flows[f"copilot:{device_code}"] = {"device_code": device_code}

        return {
            "url": verification_uri,
            "instructions": f"Enter code: {user_code}",
            "method": "device_code",
            "user_code": user_code,
            "device_code": device_code,
        }

    raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")


@router.post("/provider/{provider_id}/oauth/callback")
async def oauth_callback(
    provider_id: str,
    state: StateDep,
    code: str | None = None,
    device_code: str | None = None,
    verifier: str | None = None,
) -> dict[str, Any]:
    """Handle OAuth callback/code exchange.

    For Anthropic: exchanges authorization code for tokens.
    For Copilot: polls for token using device code.
    """
    _ = state

    if provider_id == "anthropic":
        if not code or not verifier:
            raise HTTPException(
                status_code=400, detail="Missing code or verifier for Anthropic OAuth"
            )

        try:
            token = exchange_code_for_token(code, verifier)
            store = AnthropicTokenStore()
            store.save(token)
            # Clean up flow state
            _oauth_flows.pop(f"anthropic:{verifier}", None)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        else:
            return {
                "type": "success",
                "access": token.access_token,
                "refresh": token.refresh_token,
                "expires": token.expires_at,
            }

    if provider_id == "copilot":
        if not device_code:
            raise HTTPException(status_code=400, detail="Missing device_code for Copilot OAuth")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={
                    "accept": "application/json",
                    "editor-version": "Neovim/0.6.1",
                    "editor-plugin-version": "copilot.vim/1.16.0",
                    "content-type": "application/json",
                    "user-agent": "GithubCopilot/1.155.0",
                },
                json={
                    "client_id": "Iv1.b507a08c87ecfe98",
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            data = resp.json()

        if "error" in data:
            if data["error"] == "authorization_pending":
                return {"type": "pending", "message": "Waiting for user authorization"}
            detail = data.get("error_description", data["error"])
            raise HTTPException(status_code=400, detail=detail)

        if access_token := data.get("access_token"):
            # Clean up flow state
            _oauth_flows.pop(f"copilot:{device_code}", None)
            return {
                "type": "success",
                "access": access_token,
                "refresh": data.get("refresh_token"),
                "expires": None,  # Copilot tokens don't expire the same way
            }

        return {"type": "pending", "message": "No token received yet"}
    raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
