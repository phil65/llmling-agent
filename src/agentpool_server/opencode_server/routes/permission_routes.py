"""Permission routes for OpenCode TUI compatibility."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentpool import log
from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models.events import PermissionResolvedEvent
from agentpool_server.opencode_server.routes.session_routes import PermissionResponse


router = APIRouter(prefix="/permission", tags=["permission"])
logger = log.get_logger(__name__)


@router.post("/{permission_id}/reply")
async def reply_to_permission(
    permission_id: str,
    body: PermissionResponse,
    state: StateDep,
) -> bool:
    """Respond to a pending permission request (OpenCode TUI compatibility).

    This endpoint handles the OpenCode TUI's expected format:
    POST /permission/{permission_id}/reply

    The response can be:
    - "once": Allow this tool execution once
    - "always": Always allow this tool (remembered for session)
    - "reject": Reject this tool execution
    """
    logger.info("received reply", reply=body.reply, permission_id=permission_id)
    # Find which session has this permission request
    for session_id, input_provider in state.input_providers.items():
        # Check if this permission belongs to this session
        if permission_id not in input_provider._pending_permissions:
            continue
        # Resolve the permission
        resolved = input_provider.resolve_permission(permission_id, body.reply)
        logger.info("Resolved permission", resolved=resolved)
        if not resolved:
            detail = "Permission not found or already resolved"
            raise HTTPException(status_code=404, detail=detail)
        event = PermissionResolvedEvent.create(
            session_id=session_id,
            request_id=permission_id,
            reply=body.reply,
        )
        await state.broadcast_event(event)
        return True

    # Permission not found in any session
    raise HTTPException(status_code=404, detail="Permission not found")
