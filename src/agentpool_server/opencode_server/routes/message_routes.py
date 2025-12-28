"""Message routes."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Query

from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (  # noqa: TC001
    AssistantMessage,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageWithParts,
    Part,
    SessionStatus,
    TextPart,
    TimeCreated,
    TimeCreatedUpdated,
    TimeStartEnd,
    Tokens,
    TokensCache,
    UserMessage,
)


router = APIRouter(prefix="/session/{session_id}", tags=["message"])


@router.get("/message")
async def list_messages(
    session_id: str,
    state: StateDep,
    limit: int | None = Query(default=None),
) -> list[MessageWithParts]:
    """List messages in a session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = state.messages.get(session_id, [])
    if limit:
        messages = messages[-limit:]
    return messages


@router.post("/message")
async def send_message(
    session_id: str,
    request: MessageRequest,
    state: StateDep,
) -> MessageWithParts:
    """Send a message and get response.

    TODO: Integrate with AgentPool agents.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    now = time.time()

    # Create user message
    user_msg_id = request.message_id or str(uuid.uuid4())
    user_message = UserMessage(
        id=user_msg_id,
        session_id=session_id,
        time=TimeCreated(created=now),
    )

    # Create parts from request
    user_parts: list[Part] = []
    for i, part in enumerate(request.parts):
        if part.type == "text":
            user_parts.append(
                TextPart(
                    id=f"{user_msg_id}-{i}",
                    message_id=user_msg_id,
                    session_id=session_id,
                    text=part.text,
                    time=TimeStartEnd(start=now, end=now),
                )
            )

    user_msg_with_parts = MessageWithParts(info=user_message, parts=user_parts)
    state.messages[session_id].append(user_msg_with_parts)

    # Mark session as running
    state.session_status[session_id] = SessionStatus(running=True)

    # TODO: Actually call the agent here
    # For now, return a placeholder response
    assistant_msg_id = str(uuid.uuid4())
    response_time = time.time()

    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        model_id=request.model_id,
        provider_id=request.provider_id,
        mode=request.mode or "default",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        system=[],
        time=MessageTime(created=now, completed=response_time),
        tokens=Tokens(
            cache=TokensCache(read=0, write=0),
            input=0,
            output=0,
            reasoning=0,
        ),
        cost=0.0,
    )

    # Placeholder response
    response_parts: list[Part] = [
        TextPart(
            id=f"{assistant_msg_id}-0",
            message_id=assistant_msg_id,
            session_id=session_id,
            text="[OpenCode server placeholder] Agent integration pending.",
            time=TimeStartEnd(start=now, end=response_time),
        )
    ]

    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=response_parts)
    state.messages[session_id].append(assistant_msg_with_parts)

    # Mark session as not running
    state.session_status[session_id] = SessionStatus(running=False)

    # Update session timestamp
    session = state.sessions[session_id]
    state.sessions[session_id] = session.model_copy(
        update={"time": TimeCreatedUpdated(created=session.time.created, updated=response_time)}
    )

    return assistant_msg_with_parts


@router.get("/message/{message_id}")
async def get_message(
    session_id: str,
    message_id: str,
    state: StateDep,
) -> MessageWithParts:
    """Get a specific message."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    for msg in state.messages.get(session_id, []):
        if msg.info.id == message_id:
            return msg

    raise HTTPException(status_code=404, detail="Message not found")
