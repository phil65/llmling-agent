"""LSP (Language Server Protocol) routes.

Provides endpoints for LSP server status and diagnostics,
compatible with OpenCode's LSP API.
"""

from __future__ import annotations

from contextlib import suppress
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models.events import LspStatus, LspUpdatedEvent


# =============================================================================
# Diagnostic Models (matching OpenCode's LSP diagnostic format)
# =============================================================================


class DiagnosticPosition(BaseModel):
    """Position in a text document."""

    line: int
    character: int


class DiagnosticRange(BaseModel):
    """Range in a text document."""

    start: DiagnosticPosition
    end: DiagnosticPosition


class Diagnostic(BaseModel):
    """LSP Diagnostic matching vscode-languageserver-types format."""

    range: DiagnosticRange
    message: str
    severity: int | None = None  # 1=Error, 2=Warning, 3=Info, 4=Hint
    code: str | int | None = None
    source: str | None = None


class FormatterStatus(BaseModel):
    """Formatter status information."""

    id: str
    """Formatter identifier."""

    name: str
    """Formatter name."""

    root: str
    """Workspace root path."""

    status: Literal["connected", "error"]
    """Connection status."""


router = APIRouter(tags=["lsp"])


@router.get("/lsp")
async def list_lsp_servers(state: StateDep) -> list[LspStatus]:
    """List all active LSP servers.

    Returns the status of all running LSP servers, including their
    connection state and workspace root.

    Returns:
        List of LSP server status objects.
    """
    servers: list[LspStatus] = []
    for server_id, server_state in state.lsp_manager._servers.items():
        # Get relative root path
        root_uri = server_state.root_uri or ""
        if root_uri.startswith("file://"):
            root_path = root_uri[7:]  # Remove file:// prefix
            # Make path relative to working directory
            with suppress(ValueError):
                root_path = os.path.relpath(root_path, state.working_dir)
        else:
            root_path = root_uri
        status = "connected" if server_state.initialized else "error"
        servers.append(LspStatus(id=server_id, name=server_id, root=root_path, status=status))
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
    # Default to working directory if no root provided
    if root_uri is None:
        root_uri = f"file://{state.working_dir}"

    try:
        server_state = await state.lsp_manager.start_server(server_id, root_uri)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Emit lsp.updated event to notify clients of server status change
    await state.broadcast_event(LspUpdatedEvent())
    # Get relative root path for response
    root_path = root_uri
    if root_uri.startswith("file://"):
        root_path = root_uri[7:]
        with suppress(ValueError):
            root_path = os.path.relpath(root_path, state.working_dir)
    status = "connected" if server_state.initialized else "error"
    return LspStatus(id=server_id, name=server_id, root=root_path, status=status)


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
    await state.lsp_manager.stop_server(server_id)
    # Emit lsp.updated event to notify clients of server status change
    await state.broadcast_event(LspUpdatedEvent())
    return {"status": "ok", "message": f"Server {server_id} stopped"}


@router.get("/lsp/diagnostics")
async def get_diagnostics(
    state: StateDep,
    path: str | None = Query(None, description="File path to get diagnostics for"),
) -> dict[str, list[Diagnostic]]:
    """Get diagnostics from all active LSP servers.

    Returns diagnostics organized by file path. If a specific path is provided,
    returns diagnostics only for that file using CLI diagnostics.

    This uses CLI-based diagnostic tools (pyright, mypy, etc.) which are more
    reliable for on-demand checks than the LSP push model.

    Args:
        state: Server state dependency (injected).
        path: Optional file path to get diagnostics for.

    Returns:
        Dictionary mapping file paths to lists of diagnostic objects.
    """
    results: dict[str, list[Diagnostic]] = {}

    # If a specific path is provided, run CLI diagnostics for it
    if path:
        # Make path absolute if needed
        if not os.path.isabs(path):  # noqa: PTH117
            path = os.path.join(state.working_dir, path)  # noqa: PTH118

        # Find the appropriate server for this file
        server_info = state.lsp_manager.get_server_for_file(path)
        if server_info and server_info.has_cli_diagnostics:
            try:
                result = await state.lsp_manager.run_cli_diagnostics(server_info.id, [path])
                if result.success and result.diagnostics:
                    for diag in result.diagnostics:
                        file_path = diag.file or path
                        if file_path not in results:
                            results[file_path] = []
                        # Convert from 1-based (CLI tools) to 0-based (LSP)
                        results[file_path].append(
                            Diagnostic(
                                range=DiagnosticRange(
                                    start=DiagnosticPosition(
                                        line=max(0, diag.line - 1),
                                        character=max(0, diag.column - 1),
                                    ),
                                    end=DiagnosticPosition(
                                        line=max(0, (diag.end_line or diag.line) - 1),
                                        character=max(0, (diag.end_column or diag.column) - 1),
                                    ),
                                ),
                                message=diag.message,
                                severity=_severity_to_lsp(diag.severity),
                                code=diag.code,
                                source=diag.source or server_info.id,
                            )
                        )
            except Exception:  # noqa: BLE001
                # CLI diagnostics failed, return empty
                pass

    return results


def _severity_to_lsp(severity: str) -> int:
    """Convert severity string to LSP severity number."""
    mapping = {
        "error": 1,
        "warning": 2,
        "info": 3,
        "hint": 4,
    }
    return mapping.get(severity.lower(), 1)


@router.get("/lsp/servers")
async def list_available_servers(state: StateDep) -> list[dict[str, object]]:
    """List all registered (available) LSP servers.

    Returns information about all LSP servers that can be started,
    regardless of whether they are currently running.

    Returns:
        List of server configurations.
    """
    servers = []
    for server_id, config in state.lsp_manager._server_configs.items():
        servers.append({
            "id": server_id,
            "extensions": config.extensions,
            "running": server_id in state.lsp_manager._servers,
        })

    return servers


# =============================================================================
# Formatter Routes
# =============================================================================


@router.get("/formatter")
async def list_formatters(state: StateDep) -> list[FormatterStatus]:
    """List all active formatters.

    Returns the status of all running formatters, including their
    connection state and workspace root.

    Note: This is currently a stub that returns an empty list.
    Formatter support can be added in the future.

    Returns:
        List of formatter status objects.
    """
    # Stub implementation - formatters not yet implemented
    # OpenCode has formatters like prettier, biome, etc.
    # For now, return empty list
    _ = state  # Reserved for future use
    return []
