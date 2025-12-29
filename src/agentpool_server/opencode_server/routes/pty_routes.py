"""PTY (Pseudo-Terminal) routes."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, WebSocketDisconnect

from agentpool_server.opencode_server.models import (
    PtyInfo,
)


if TYPE_CHECKING:
    from fastapi import WebSocket

    from agentpool_server.opencode_server.dependencies import StateDep
    from agentpool_server.opencode_server.models import (
        PtyCreateRequest,
        PtyUpdateRequest,
    )


router = APIRouter(prefix="/pty", tags=["pty"])


@dataclass
class PtySession:
    """Active PTY session."""

    info: PtyInfo
    process: asyncio.subprocess.Process | None = None
    buffer: str = ""
    subscribers: set[WebSocket] = field(default_factory=set)
    read_task: asyncio.Task[Any] | None = None


# Global PTY sessions store (could be moved to ServerState later)
_pty_sessions: dict[str, PtySession] = {}


def _generate_pty_id() -> str:
    """Generate a unique PTY ID."""
    import uuid

    return f"pty_{uuid.uuid4().hex[:12]}"


@router.get("")
async def list_ptys(state: StateDep) -> list[PtyInfo]:
    """List all PTY sessions."""
    return [session.info for session in _pty_sessions.values()]


@router.post("")
async def create_pty(request: PtyCreateRequest, state: StateDep) -> PtyInfo:
    """Create a new PTY session."""
    pty_id = _generate_pty_id()

    # Determine shell command
    command = request.command or os.environ.get("SHELL", "/bin/bash")
    args = request.args or []
    if command.endswith("sh") and "-l" not in args:
        args = ["-l", *args]

    cwd = request.cwd or state.working_dir
    title = request.title or f"Terminal {pty_id[-4:]}"

    # Build environment
    env = {**os.environ, **(request.env or {}), "TERM": "xterm-256color"}

    # Start the process
    try:
        full_command = [command, *args]
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create PTY: {e}") from e

    info = PtyInfo(
        id=pty_id,
        title=title,
        command=command,
        args=args,
        cwd=cwd,
        status="running",
        pid=process.pid or 0,
    )

    session = PtySession(info=info, process=process)
    _pty_sessions[pty_id] = session

    # Start background task to read output
    session.read_task = asyncio.create_task(_read_pty_output(pty_id))

    # TODO: Broadcast pty.created event

    return info


async def _read_pty_output(pty_id: str) -> None:
    """Background task to read PTY output and distribute to subscribers."""
    session = _pty_sessions.get(pty_id)
    if not session or not session.process or not session.process.stdout:
        return

    try:
        while True:
            data = await session.process.stdout.read(4096)
            if not data:
                break

            decoded = data.decode("utf-8", errors="replace")

            if session.subscribers:
                # Send to all connected WebSocket clients
                disconnected = set()
                for ws in session.subscribers:
                    try:
                        await ws.send_text(decoded)
                    except Exception:  # noqa: BLE001
                        disconnected.add(ws)
                session.subscribers -= disconnected
            else:
                # Buffer output if no subscribers
                session.buffer += decoded
                # Limit buffer size
                if len(session.buffer) > 100000:  # noqa: PLR2004
                    session.buffer = session.buffer[-50000:]

    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        # Mark as exited
        if pty_id in _pty_sessions:
            session = _pty_sessions[pty_id]
            session.info.status = "exited"
            # TODO: Broadcast pty.exited event


@router.get("/{pty_id}")
async def get_pty(pty_id: str, state: StateDep) -> PtyInfo:
    """Get PTY session details."""
    session = _pty_sessions.get(pty_id)
    if not session:
        raise HTTPException(status_code=404, detail="PTY session not found")
    return session.info


@router.patch("/{pty_id}")
async def update_pty(pty_id: str, request: PtyUpdateRequest, state: StateDep) -> PtyInfo:
    """Update PTY session (title, resize)."""
    session = _pty_sessions.get(pty_id)
    if not session:
        raise HTTPException(status_code=404, detail="PTY session not found")

    if request.title:
        session.info.title = request.title

    # Note: resize requires actual PTY support (not available with subprocess)
    # Real implementation would use bun-pty or similar

    # TODO: Broadcast pty.updated event

    return session.info


@router.delete("/{pty_id}")
async def remove_pty(pty_id: str, state: StateDep) -> dict[str, bool]:
    """Remove/kill PTY session."""
    session = _pty_sessions.get(pty_id)
    if not session:
        raise HTTPException(status_code=404, detail="PTY session not found")

    # Kill the process
    if session.process and session.process.returncode is None:
        try:
            session.process.terminate()
            await asyncio.wait_for(session.process.wait(), timeout=5.0)
        except TimeoutError:
            session.process.kill()
        except Exception:  # noqa: BLE001
            pass

    # Cancel read task
    if session.read_task and not session.read_task.done():
        session.read_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session.read_task

    # Close all WebSocket connections
    for ws in session.subscribers:
        with contextlib.suppress(Exception):
            await ws.close()

    del _pty_sessions[pty_id]

    # TODO: Broadcast pty.deleted event

    return {"success": True}


@router.websocket("/{pty_id}/connect")
async def connect_pty(websocket: WebSocket, pty_id: str) -> None:
    """Connect to PTY via WebSocket."""
    session = _pty_sessions.get(pty_id)
    if not session:
        await websocket.close(code=4004, reason="PTY session not found")
        return

    await websocket.accept()
    session.subscribers.add(websocket)

    # Send buffered output
    if session.buffer:
        try:
            await websocket.send_text(session.buffer)
            session.buffer = ""
        except Exception:  # noqa: BLE001
            pass

    try:
        while True:
            # Receive input from client
            data = await websocket.receive_text()

            # Write to PTY stdin
            if session.process and session.process.stdin and session.info.status == "running":
                try:
                    session.process.stdin.write(data.encode())
                    await session.process.stdin.drain()
                except Exception:  # noqa: BLE001
                    break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        session.subscribers.discard(websocket)
