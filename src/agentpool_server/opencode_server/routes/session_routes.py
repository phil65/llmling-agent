"""Session routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentpool_server.opencode_server import identifier
from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (  # noqa: TC001
    Session,
    SessionCreatedEvent,
    SessionCreateRequest,
    SessionDeletedEvent,
    SessionDeletedProperties,
    SessionInfoProperties,
    SessionStatus,
    SessionUpdatedEvent,
    SessionUpdateRequest,
    TimeCreatedUpdated,
    Todo,
)
from agentpool_server.opencode_server.time_utils import now_ms


router = APIRouter(prefix="/session", tags=["session"])


@router.get("")
async def list_sessions(state: StateDep) -> list[Session]:
    """List all sessions."""
    return list(state.sessions.values())


@router.post("")
async def create_session(
    state: StateDep,
    request: SessionCreateRequest | None = None,
) -> Session:
    """Create a new session."""
    now = now_ms()
    session_id = identifier.ascending("session")
    session = Session(
        id=session_id,
        project_id="default",  # TODO: Get from config/request
        directory=state.working_dir,
        title=request.title if request and request.title else "New Session",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
        parent_id=request.parent_id if request else None,
    )
    state.sessions[session_id] = session
    state.messages[session_id] = []
    state.session_status[session_id] = SessionStatus(type="idle")
    state.todos[session_id] = []

    await state.broadcast_event(SessionCreatedEvent(properties=SessionInfoProperties(info=session)))

    return session


@router.get("/status")
async def get_session_status(state: StateDep) -> dict[str, SessionStatus]:
    """Get status for all sessions.

    Returns only non-idle sessions. If all sessions are idle, returns empty dict.
    """
    return {sid: status for sid, status in state.session_status.items() if status.type != "idle"}


@router.get("/{session_id}")
async def get_session(session_id: str, state: StateDep) -> Session:
    """Get session details."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.sessions[session_id]


@router.patch("/{session_id}")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    state: StateDep,
) -> Session:
    """Update session properties."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = state.sessions[session_id]
    if request.title is not None:
        session = session.model_copy(
            update={
                "title": request.title,
                "time": TimeCreatedUpdated(
                    created=session.time.created,
                    updated=now_ms(),
                ),
            }
        )
    state.sessions[session_id] = session

    await state.broadcast_event(SessionUpdatedEvent(properties=SessionInfoProperties(info=session)))

    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str, state: StateDep) -> bool:
    """Delete a session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del state.sessions[session_id]
    state.messages.pop(session_id, None)
    state.session_status.pop(session_id, None)
    state.todos.pop(session_id, None)

    await state.broadcast_event(
        SessionDeletedEvent(properties=SessionDeletedProperties(session_id=session_id))
    )

    return True


@router.post("/{session_id}/abort")
async def abort_session(session_id: str, state: StateDep) -> bool:
    """Abort a running session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    # TODO: Actually abort running operations
    state.session_status[session_id] = SessionStatus(type="idle")
    return True


@router.get("/{session_id}/todo")
async def get_session_todos(session_id: str, state: StateDep) -> list[Todo]:
    """Get todos for a session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.todos.get(session_id, [])


@router.get("/{session_id}/diff")
async def get_session_diff(
    session_id: str,
    state: StateDep,
    message_id: str | None = None,
) -> list[dict[str, str]]:
    """Get file diffs for a session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    # TODO: Track and return actual file diffs
    return []
