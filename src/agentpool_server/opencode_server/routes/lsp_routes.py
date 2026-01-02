"""LSP (Language Server Protocol) routes.

Provides endpoints for LSP server status and diagnostics,
compatible with OpenCode's LSP API.
"""

from __future__ import annotations

from contextlib import suppress
import os

from fastapi import APIRouter, HTTPException, Query

from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models.events import LspStatus, LspUpdatedEvent


router = APIRouter(tags=["lsp"])


@router.get("/lsp")
async def list_lsp_servers(state: StateDep) -> list[LspStatus]:
    """List all active LSP servers.

    Returns the status of all running LSP servers, including their
    connection state and workspace root.

    Returns:
        List of LSP server status objects.
    """
    try:
        lsp_manager = state.get_or_create_lsp_manager()
    except RuntimeError:
        # Agent doesn't have an execution environment - return empty list
        return []

    servers: list[LspStatus] = []
    for server_id, server_state in lsp_manager._servers.items():
        # Get relative root path
        root_uri = server_state.root_uri or ""
        if root_uri.startswith("file://"):
            root_path = root_uri[7:]  # Remove file:// prefix
            # Make path relative to working directory
            with suppress(ValueError):
                root_path = os.path.relpath(root_path, state.working_dir)
        else:
            root_path = root_uri

        servers.append(
            LspStatus(
                id=server_id,
                name=server_id,
                root=root_path,
                status="connected" if server_state.initialized else "error",
            )
        )

    return servers


@router.post("/lsp/start")
async def start_lsp_server(
    state: StateDep,
    server_id: str = Query(..., description="LSP server ID (e.g., 'pyright', 'rust-analyzer')"),
    root_uri: str | None = Query(None, description="Workspace root URI"),
) -> LspStatus:
    """Start an LSP server.

    Starts the specified LSP server for the given workspace root.
    If no root_uri is provided, uses the server's working directory.

    Args:
        state: Server state dependency (injected).
        server_id: The LSP server identifier (e.g., 'pyright', 'typescript').
        root_uri: Optional workspace root URI (file:// format).

    Returns:
        The started server's status.

    Raises:
        HTTPException: If the server fails to start or is not registered.
    """
    try:
        lsp_manager = state.get_or_create_lsp_manager()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Default to working directory if no root provided
    if root_uri is None:
        root_uri = f"file://{state.working_dir}"

    try:
        server_state = await lsp_manager.start_server(server_id, root_uri)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Emit lsp.updated event to notify clients of server status change
    await state.broadcast_event(LspUpdatedEvent.create())

    # Get relative root path for response
    root_path = root_uri
    if root_uri.startswith("file://"):
        root_path = root_uri[7:]
        with suppress(ValueError):
            root_path = os.path.relpath(root_path, state.working_dir)

    return LspStatus(
        id=server_id,
        name=server_id,
        root=root_path,
        status="connected" if server_state.initialized else "error",
    )


@router.post("/lsp/stop")
async def stop_lsp_server(
    state: StateDep,
    server_id: str = Query(..., description="LSP server ID to stop"),
) -> dict[str, str]:
    """Stop an LSP server.

    Args:
        state: Server state dependency (injected).
        server_id: The LSP server identifier to stop.

    Returns:
        Success message.
    """
    try:
        lsp_manager = state.get_or_create_lsp_manager()
    except RuntimeError:
        return {"status": "ok", "message": "No LSP manager active"}

    await lsp_manager.stop_server(server_id)

    # Emit lsp.updated event to notify clients of server status change
    await state.broadcast_event(LspUpdatedEvent.create())

    return {"status": "ok", "message": f"Server {server_id} stopped"}


@router.get("/lsp/diagnostics")
async def get_diagnostics(
    state: StateDep,
    path: str | None = Query(None, description="File path to get diagnostics for"),
) -> dict[str, list[dict[str, object]]]:
    """Get diagnostics from all active LSP servers.

    Returns diagnostics organized by file path. If a specific path is provided,
    returns diagnostics only for that file.

    Args:
        state: Server state dependency (injected).
        path: Optional file path to filter diagnostics.

    Returns:
        Dictionary mapping file paths to lists of diagnostic objects.
    """
    # Validate we have LSP capability
    try:
        _ = state.get_or_create_lsp_manager()
    except RuntimeError:
        return {}

    # Collect diagnostics from all servers
    # Note: Full LSP diagnostics require the server to have processed files
    # via textDocument/didOpen. For now, return empty dict as diagnostics
    # are pushed via events, not pulled.
    #
    # In the future, this could aggregate diagnostics from CLI tools
    # using run_cli_diagnostics() for on-demand checks.
    _ = path  # Reserved for future filtering

    return {}


@router.get("/lsp/servers")
async def list_available_servers(state: StateDep) -> list[dict[str, object]]:
    """List all registered (available) LSP servers.

    Returns information about all LSP servers that can be started,
    regardless of whether they are currently running.

    Returns:
        List of server configurations.
    """
    try:
        lsp_manager = state.get_or_create_lsp_manager()
    except RuntimeError:
        return []

    servers = []
    for server_id, config in lsp_manager._server_configs.items():
        servers.append({
            "id": server_id,
            "extensions": config.extensions,
            "running": server_id in lsp_manager._servers,
        })

    return servers
