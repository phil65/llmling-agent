"""Message routes."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, status

from agentpool.log import get_logger
from agentpool.utils import identifiers as identifier
from agentpool.utils.time_utils import now_ms
from agentpool_server.opencode_server.converters import (
    extract_user_prompt_from_parts,
    opencode_to_chat_message,
)
from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import (
    AssistantMessage,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
    PartUpdatedEvent,
    SessionIdleEvent,
    SessionStatus,
    SessionStatusEvent,
    StepStartPart,
    TextPart,
    TimeCreated,
    TimeCreatedUpdated,
    Tokens,
    TokensCache,
    UserMessage,
)
from agentpool_server.opencode_server.models.events import LspUpdatedEvent
from agentpool_server.opencode_server.models.message import UserMessageModel
from agentpool_server.opencode_server.routes.session_routes import get_or_load_session
from agentpool_server.opencode_server.stream_adapter import OpenCodeStreamAdapter


if TYPE_CHECKING:
    from agentpool_server.opencode_server.models import Part
    from agentpool_server.opencode_server.state import ServerState


logger = get_logger(__name__)


def _warmup_lsp_for_files(state: ServerState, file_paths: list[str]) -> None:
    """Warm up LSP servers for the given file paths.

    This starts LSP servers asynchronously based on file extensions.
    Like OpenCode's LSP.touchFile(), this triggers server startup without waiting.

    Args:
        state: Server state with LSP manager
        file_paths: List of file paths that were accessed
    """
    logger.info("_warmup_lsp_for_files called with", file_paths=file_paths)
    lsp_manager = state.lsp_manager

    async def warmup_files() -> None:
        """Start LSP servers for each file path."""
        logger.info("warmup_files task started")

        servers_started = False
        for path in file_paths:
            # Find appropriate server for this file
            server_info = lsp_manager.get_server_for_file(path)
            if server_info is None:
                continue
            server_id = server_info.id
            if lsp_manager.is_running(server_id):
                logger.info("Server with same id already running", server_id=server_id)
                continue

            # Start server for workspace root
            root_uri = f"file://{state.working_dir}"
            logger.info("Starting server...", server_id=server_id)
            try:
                await lsp_manager.start_server(server_id, root_uri)
                servers_started = True
                logger.info("Server started successfully", server_id=server_id)
            except Exception as e:  # noqa: BLE001
                # Don't fail on LSP startup errors
                logger.info("Failed to start server", error=e, server_id=server_id)

        # Emit lsp.updated event if any servers started
        if servers_started:
            logger.info("Broadcasting LspUpdatedEvent")
            await state.broadcast_event(LspUpdatedEvent())
        logger.info("warmup_files task completed")

    # Run warmup in background (don't block the event handler)
    logger.info("Creating background task for warmup")
    state.create_background_task(warmup_files(), name="lsp-warmup")


async def persist_message_to_storage(
    state: ServerState,
    msg: MessageWithParts,
    session_id: str,
) -> None:
    """Persist an OpenCode message to storage.

    Converts the OpenCode MessageWithParts to ChatMessage and saves it.

    Args:
        state: Server state with pool reference
        msg: OpenCode message to persist
        session_id: Session/conversation ID
    """
    if state.pool.storage is None:
        return

    chat_msg = opencode_to_chat_message(msg, session_id=session_id)
    with contextlib.suppress(Exception):
        await state.pool.storage.log_message(chat_msg)


router = APIRouter(prefix="/session/{session_id}", tags=["message"])


@router.get("/message")
async def list_messages(
    session_id: str,
    state: StateDep,
    limit: int | None = Query(default=None),
) -> list[MessageWithParts]:
    """List messages in a session."""
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = state.messages.get(session_id, [])
    return messages[-limit:] if limit else messages


async def _process_message(
    session_id: str,
    request: MessageRequest,
    state: StateDep,
) -> MessageWithParts:
    """Internal helper to process a message request.

    This does the actual work of creating messages, running the agent,
    and broadcasting events. Used by both sync and async endpoints.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # --- Create user message ---
    user_msg_id = identifier.ascending("message", request.message_id)
    model = UserMessageModel(
        provider_id=request.model.provider_id if request.model else "agentpool",
        model_id=request.model.model_id if request.model else "default",
    )
    user_message = UserMessage(
        id=user_msg_id,
        session_id=session_id,
        time=TimeCreated.now(),
        agent=request.agent or "default",
        model=model,
        variant=request.variant,
    )

    user_parts: list[Part] = [
        TextPart(
            id=identifier.ascending("part"),
            message_id=user_msg_id,
            session_id=session_id,
            text=part.text,
        )
        for part in request.parts
        if part.type == "text"
    ]
    user_msg_with_parts = MessageWithParts(info=user_message, parts=user_parts)
    state.messages[session_id].append(user_msg_with_parts)
    await persist_message_to_storage(state, user_msg_with_parts, session_id)
    await state.broadcast_event(MessageUpdatedEvent.create(user_message))
    for part in user_parts:
        await state.broadcast_event(PartUpdatedEvent.create(part))
    # --- Mark session busy ---
    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="busy")))
    # --- Extract user prompt ---
    part_data = [p.model_dump() for p in request.parts]
    user_prompt = extract_user_prompt_from_parts(part_data, fs=state.fs)
    # --- Create assistant message ---
    assistant_msg_id = identifier.ascending("message")
    now = now_ms()
    tokens = Tokens(cache=TokensCache(read=0, write=0))
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id=user_msg_id,
        model_id=request.model.model_id if request.model else "default",
        provider_id=request.model.provider_id if request.model else "agentpool",
        mode=request.agent or "default",
        agent=request.agent or "default",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now, completed=None),
        tokens=tokens,
        cost=0.0,
    )
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
    # Step-start part
    part_id = identifier.ascending("part")
    step_start = StepStartPart(id=part_id, message_id=assistant_msg_id, session_id=session_id)
    assistant_msg_with_parts.parts.append(step_start)
    await state.broadcast_event(PartUpdatedEvent.create(step_start))
    # --- Resolve agent and variant ---
    agent = state.agent
    if request.agent and state.agent.agent_pool is not None:
        agent = state.agent.agent_pool.all_agents.get(request.agent, state.agent)
    if request.variant:
        with contextlib.suppress(Exception):
            await agent.set_mode(request.variant, category_id="thought_level")

    # --- Stream via adapter ---
    adapter = OpenCodeStreamAdapter(
        session_id=session_id,
        assistant_msg_id=assistant_msg_id,
        assistant_msg=assistant_msg_with_parts,
        working_dir=state.working_dir,
        on_file_paths=lambda paths: _warmup_lsp_for_files(state, paths),
    )
    iterator = agent.run_stream(user_prompt, session_id=session_id)
    async for oc_event in adapter.process_stream(iterator):
        await state.broadcast_event(oc_event)

    for oc_event in adapter.finalize():
        await state.broadcast_event(oc_event)

    # --- Finalize assistant message ---
    response_time = now_ms()
    preview = adapter.response_text[:100] if adapter.response_text else "EMPTY"
    logger.info("Response text", text_preview=preview)
    updated_assistant = assistant_message.model_copy(
        update={
            "time": MessageTime(created=now, completed=response_time),
            "tokens": Tokens(
                cache=TokensCache(read=0, write=0),
                input=adapter.input_tokens,
                output=adapter.output_tokens,
            ),
            "cost": adapter.total_cost,
        }
    )
    assistant_msg_with_parts.info = updated_assistant
    await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
    await persist_message_to_storage(state, assistant_msg_with_parts, session_id)
    # --- Mark session idle ---
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    await state.broadcast_event(SessionIdleEvent.create(session_id))
    # --- Update session timestamp ---
    session = state.sessions[session_id]
    state.sessions[session_id] = session.model_copy(
        update={"time": TimeCreatedUpdated(created=session.time.created, updated=response_time)}
    )
    return assistant_msg_with_parts


@router.post("/message")
async def send_message(
    session_id: str,
    request: MessageRequest,
    state: StateDep,
) -> MessageWithParts:
    """Send a message and wait for the agent's response.

    This is the synchronous version - waits for completion before returning.
    For async processing, use POST /session/{id}/prompt_async instead.
    """
    return await _process_message(session_id, request, state)


@router.post("/prompt_async", status_code=status.HTTP_204_NO_CONTENT)
async def send_message_async(session_id: str, request: MessageRequest, state: StateDep) -> None:
    """Send a message asynchronously without waiting for response.

    Starts the agent processing in the background and returns immediately.
    Client should listen to SSE events to get updates.

    Returns 204 No Content immediately.
    """
    # Create background task to process the message
    state.create_background_task(
        _process_message(session_id, request, state),
        name=f"process_message_{session_id}",
    )


@router.get("/message/{message_id}")
async def get_message(
    session_id: str,
    message_id: str,
    state: StateDep,
) -> MessageWithParts:
    """Get a specific message."""
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    for msg in state.messages.get(session_id, []):
        if msg.info.id == message_id:
            return msg

    raise HTTPException(status_code=404, detail="Message not found")
