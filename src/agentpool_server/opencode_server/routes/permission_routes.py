"""Permission routes for OpenCode TUI compatibility."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models.events import PermissionResolvedEvent
from agentpool_server.opencode_server.routes.session_routes import PermissionResponse


router = APIRouter(prefix="/permission", tags=["permission"])


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
    print(f"DEBUG permission endpoint: received reply '{body.reply}' for perm_id={permission_id}")
    print(f"DEBUG permission endpoint: searching in {len(state.input_providers)} sessions")
    # Find which session has this permission request
    for session_id, input_provider in state.input_providers.items():
        pending_perms = list(input_provider._pending_permissions.keys())
        print(
            f"DEBUG permission endpoint: session {session_id} has "
            f"{len(pending_perms)} pending: {pending_perms}"
        )
        # Check if this permission belongs to this session
        if permission_id in input_provider._pending_permissions:
            print(f"DEBUG permission endpoint: found permission in session {session_id}")
            # Resolve the permission
            resolved = input_provider.resolve_permission(permission_id, body.reply)
            print(f"DEBUG permission endpoint: resolve_permission returned {resolved}")
            if not resolved:
                raise HTTPException(
                    status_code=404,
                    detail="Permission not found or already resolved",
                )

            await state.broadcast_event(
                PermissionResolvedEvent.create(
                    session_id=session_id,
                    request_id=permission_id,
                    reply=body.reply,
                )
            )

            return True

    # Permission not found in any session
    raise HTTPException(status_code=404, detail="Permission not found")
