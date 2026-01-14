"""Message routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic_ai import FunctionToolCallEvent
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart as PydanticTextPart,
    TextPartDelta,
    ToolCallPart as PydanticToolCallPart,
)

from agentpool.agents.events import (
    CompactionEvent,
    FileContentItem,
    LocationContentItem,
    StreamCompleteEvent,
    SubAgentEvent,
    TextContentItem,
    ToolCallCompleteEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from agentpool.agents.events.infer_info import derive_rich_tool_info
from agentpool.utils import identifiers as identifier
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict
from agentpool_server.opencode_server.converters import (
    _convert_params_for_ui,
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
    SessionCompactedEvent,
    SessionErrorEvent,
    SessionIdleEvent,
    SessionStatus,
    SessionStatusEvent,
    StepFinishPart,
    StepStartPart,
    TextPart,
    TimeCreated,
    TimeCreatedUpdated,
    TimeStartEnd,
    Tokens,
    TokensCache,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
    UserMessage,
)
from agentpool_server.opencode_server.models.message import UserMessageModel
from agentpool_server.opencode_server.models.parts import (
    StepFinishTokens,
    TimeStart,
    TimeStartEndCompacted,
    TimeStartEndOptional,
    TokenCache,
)
from agentpool_server.opencode_server.routes.session_routes import get_or_load_session
from agentpool_server.opencode_server.time_utils import now_ms


if TYPE_CHECKING:
    from agentpool_server.opencode_server.models import (
        Part,
    )
    from agentpool_server.opencode_server.state import ServerState


def _warmup_lsp_for_files(state: ServerState, file_paths: list[str]) -> None:
    """Warm up LSP servers for the given file paths.

    This starts LSP servers asynchronously based on file extensions.
    Like OpenCode's LSP.touchFile(), this triggers server startup without waiting.

    Args:
        state: Server state with LSP manager
        file_paths: List of file paths that were accessed
    """
    import logging

    logging.getLogger(__name__)
    print(f"[LSP] _warmup_lsp_for_files called with: {file_paths}")

    try:
        lsp_manager = state.get_or_create_lsp_manager()
        print("[LSP] Got LSP manager successfully")
    except RuntimeError as e:
        # No execution environment available for LSP
        print(f"[LSP] No LSP manager: {e}")
        return

    async def warmup_files() -> None:
        """Start LSP servers for each file path."""
        print("[LSP] warmup_files task started")
        from agentpool_server.opencode_server.models.events import LspUpdatedEvent

        servers_started = False
        for path in file_paths:
            # Find appropriate server for this file
            server_info = lsp_manager.get_server_for_file(path)
            print(f"[LSP] Server for {path}: {server_info.id if server_info else None}")
            if server_info is None:
                continue

            server_id = server_info.id
            if lsp_manager.is_running(server_id):
                print(f"[LSP] Server {server_id} already running")
                continue

            # Start server for workspace root
            root_uri = f"file://{state.working_dir}"
            try:
                print(f"[LSP] Starting server {server_id}...")
                await lsp_manager.start_server(server_id, root_uri)
                servers_started = True
                print(f"[LSP] Server {server_id} started successfully")
            except Exception as e:  # noqa: BLE001
                # Don't fail on LSP startup errors
                print(f"[LSP] Failed to start server {server_id}: {e}")

        # Emit lsp.updated event if any servers started
        if servers_started:
            print("[LSP] Broadcasting LspUpdatedEvent")
            await state.broadcast_event(LspUpdatedEvent.create())
        print("[LSP] warmup_files task completed")

    # Run warmup in background (don't block the event handler)
    print("[LSP] Creating background task for warmup")
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

    try:
        # Convert to ChatMessage
        chat_msg = opencode_to_chat_message(msg, conversation_id=session_id)
        # Persist via storage manager
        await state.pool.storage.log_message(chat_msg)
    except Exception:  # noqa: BLE001
        # Don't fail the request if storage fails
        pass


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
    if limit:
        messages = messages[-limit:]
    return messages


async def _process_message(  # noqa: PLR0915
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

    now = now_ms()
    # Create user message with sortable ID
    user_msg_id = identifier.ascending("message", request.message_id)
    user_message = UserMessage(
        id=user_msg_id,
        session_id=session_id,
        time=TimeCreated(created=now),
        agent=request.agent or "default",
        model=UserMessageModel(
            provider_id=request.model.provider_id if request.model else "agentpool",
            model_id=request.model.model_id if request.model else "default",
        ),
    )

    # Create parts from request
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
    # Persist user message to storage
    await persist_message_to_storage(state, user_msg_with_parts, session_id)
    # Broadcast user message created event
    await state.broadcast_event(MessageUpdatedEvent.create(user_message))
    # Broadcast user message parts so they appear in UI
    for part in user_parts:
        await state.broadcast_event(PartUpdatedEvent.create(part))
    state.session_status[session_id] = SessionStatus(type="busy")
    status_event = SessionStatusEvent.create(session_id, SessionStatus(type="busy"))
    await state.broadcast_event(status_event)
    # Extract user prompt text
    user_prompt = extract_user_prompt_from_parts([p.model_dump() for p in request.parts])
    # Create assistant message with sortable ID (must come after user message)
    assistant_msg_id = identifier.ascending("message")
    tokens = Tokens(cache=TokensCache(read=0, write=0))
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id=user_msg_id,  # Link to user message
        model_id=request.model.model_id if request.model else "default",
        provider_id=request.model.provider_id if request.model else "agentpool",
        mode=request.agent or "default",
        agent=request.agent or "default",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now, completed=None),
        tokens=tokens,
        cost=0.0,
    )
    # Initialize assistant message with empty parts
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)
    # Broadcast assistant message created
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
    # Add step-start part
    step_start = StepStartPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
    )
    assistant_msg_with_parts.parts.append(step_start)
    await state.broadcast_event(PartUpdatedEvent.create(step_start))
    # Call the agent
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    total_cost = 0.0  # Cost in dollars
    tool_parts: dict[str, ToolPart] = {}  # Track tool parts by call_id
    tool_outputs: dict[str, str] = {}  # Track accumulated output per tool call
    tool_inputs: dict[str, dict[str, Any]] = {}  # Track inputs per tool call
    # Track streaming text part for incremental updates
    text_part: TextPart | None = None
    text_part_id: str | None = None

    try:
        # Get the specified agent from the pool, or fall back to default
        agent = state.agent
        if request.agent and state.agent.agent_pool is not None:
            agent = state.agent.agent_pool.all_agents.get(request.agent, state.agent)

        # Stream events from the agent
        async for event in agent.run_stream(user_prompt, conversation_id=session_id):
            match event:
                # Text streaming start
                case PartStartEvent(part=PydanticTextPart(content=delta)):
                    response_text = delta
                    text_part_id = identifier.ascending("part")
                    text_part = TextPart(
                        id=text_part_id,
                        message_id=assistant_msg_id,
                        session_id=session_id,
                        text=delta,
                    )
                    assistant_msg_with_parts.parts.append(text_part)
                    await state.broadcast_event(PartUpdatedEvent.create(text_part, delta=delta))

                # Text streaming delta
                case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)) if delta:
                    response_text += delta
                    if text_part is not None:
                        text_part = TextPart(
                            id=text_part.id,
                            message_id=assistant_msg_id,
                            session_id=session_id,
                            text=response_text,
                        )
                        # Update in parts list
                        for i, p in enumerate(assistant_msg_with_parts.parts):
                            if isinstance(p, TextPart) and p.id == text_part.id:
                                assistant_msg_with_parts.parts[i] = text_part
                                break
                        await state.broadcast_event(PartUpdatedEvent.create(text_part, delta=delta))

                # Tool call start - from Claude Code agent or toolsets
                case ToolCallStartEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    raw_input=raw_input,
                    title=title,
                ):
                    # Convert param names for OpenCode TUI compatibility
                    ui_input = _convert_params_for_ui(raw_input) if raw_input else {}
                    if tool_call_id in tool_parts:
                        # Update existing part with the custom title
                        existing = tool_parts[tool_call_id]
                        tool_inputs[tool_call_id] = ui_input or tool_inputs.get(tool_call_id, {})

                        updated = ToolPart(
                            id=existing.id,
                            message_id=existing.message_id,
                            session_id=existing.session_id,
                            tool=existing.tool,
                            call_id=existing.call_id,
                            state=ToolStateRunning(
                                status="running",
                                time=TimeStart(start=now_ms()),
                                input=tool_inputs[tool_call_id],
                                title=title,
                            ),
                        )
                        tool_parts[tool_call_id] = updated
                        for i, p in enumerate(assistant_msg_with_parts.parts):
                            if isinstance(p, ToolPart) and p.id == existing.id:
                                assistant_msg_with_parts.parts[i] = updated
                                break
                        await state.broadcast_event(PartUpdatedEvent.create(updated))
                    else:
                        # Create new tool part with the title
                        tool_inputs[tool_call_id] = ui_input
                        tool_outputs[tool_call_id] = ""
                        tool_state = ToolStateRunning(
                            status="running",
                            time=TimeStart(start=now_ms()),
                            input=ui_input,
                            title=title,
                        )
                        tool_part = ToolPart(
                            id=identifier.ascending("part"),
                            message_id=assistant_msg_id,
                            session_id=session_id,
                            tool=tool_name,
                            call_id=tool_call_id,
                            state=tool_state,
                        )
                        tool_parts[tool_call_id] = tool_part
                        assistant_msg_with_parts.parts.append(tool_part)
                        await state.broadcast_event(PartUpdatedEvent.create(tool_part))

                # Pydantic-ai tool call events (fallback for pydantic-ai agents)
                case (
                    FunctionToolCallEvent(part=tc_part)
                    | PartStartEvent(part=PydanticToolCallPart() as tc_part)
                ) if tc_part.tool_call_id not in tool_parts:
                    tool_call_id = tc_part.tool_call_id
                    tool_name = tc_part.tool_name
                    raw_input = safe_args_as_dict(tc_part)
                    # Convert param names for OpenCode TUI compatibility
                    ui_input = _convert_params_for_ui(raw_input)
                    # Store input and initialize output accumulator
                    tool_inputs[tool_call_id] = ui_input
                    tool_outputs[tool_call_id] = ""
                    # Derive initial title; toolset events may update it later
                    rich_info = derive_rich_tool_info(tool_name, raw_input)
                    tool_state = ToolStateRunning(
                        status="running",
                        time=TimeStart(start=now_ms()),
                        input=ui_input,
                        title=rich_info.title,
                    )
                    tool_part = ToolPart(
                        id=identifier.ascending("part"),
                        message_id=assistant_msg_id,
                        session_id=session_id,
                        tool=tool_name,
                        call_id=tool_call_id,
                        state=tool_state,
                    )
                    tool_parts[tool_call_id] = tool_part
                    assistant_msg_with_parts.parts.append(tool_part)
                    await state.broadcast_event(PartUpdatedEvent.create(tool_part))

                # Tool call progress
                case ToolCallProgressEvent(
                    tool_call_id=tool_call_id,
                    title=title,
                    items=items,
                    tool_name=tool_name,
                    tool_input=event_tool_input,
                ) if tool_call_id:
                    # Extract text content from items and accumulate
                    # TODO: Handle TerminalContentItem for bash tool streaming - need to
                    # properly stream terminal output to OpenCode UI metadata
                    new_output = ""
                    file_paths: list[str] = []
                    for item in items:
                        if isinstance(item, TextContentItem):
                            new_output += item.text
                        elif isinstance(item, FileContentItem):
                            new_output += item.content
                            file_paths.append(item.path)
                        elif isinstance(item, LocationContentItem):
                            file_paths.append(item.path)

                    # Warm up LSP servers for accessed files (async, don't wait)
                    if file_paths:
                        _warmup_lsp_for_files(state, file_paths)

                    # Accumulate output (OpenCode streams via metadata.output)
                    if new_output:
                        tool_outputs[tool_call_id] = tool_outputs.get(tool_call_id, "") + new_output

                    if tool_call_id in tool_parts:
                        # Update existing part
                        existing = tool_parts[tool_call_id]
                        existing_title = getattr(existing.state, "title", "")
                        tool_input = tool_inputs.get(tool_call_id, {})
                        accumulated_output = tool_outputs.get(tool_call_id, "")
                        tool_state = ToolStateRunning(
                            status="running",
                            time=TimeStart(start=now_ms()),
                            title=title or existing_title,
                            input=tool_input,
                            metadata={"output": accumulated_output} if accumulated_output else None,
                        )
                        updated = ToolPart(
                            id=existing.id,
                            message_id=existing.message_id,
                            session_id=existing.session_id,
                            tool=existing.tool,
                            call_id=existing.call_id,
                            state=tool_state,
                        )
                        tool_parts[tool_call_id] = updated
                        for i, p in enumerate(assistant_msg_with_parts.parts):
                            if isinstance(p, ToolPart) and p.id == existing.id:
                                assistant_msg_with_parts.parts[i] = updated
                                break
                        await state.broadcast_event(PartUpdatedEvent.create(updated))
                    else:
                        # Create new tool part from progress event
                        ui_input = (
                            _convert_params_for_ui(event_tool_input) if event_tool_input else {}
                        )
                        tool_inputs[tool_call_id] = ui_input
                        accumulated_output = tool_outputs.get(tool_call_id, "")
                        tool_state = ToolStateRunning(
                            status="running",
                            time=TimeStart(start=now_ms()),
                            input=ui_input,
                            title=title or tool_name or "Running...",
                            metadata={"output": accumulated_output} if accumulated_output else None,
                        )
                        tool_part = ToolPart(
                            id=identifier.ascending("part"),
                            message_id=assistant_msg_id,
                            session_id=session_id,
                            tool=tool_name or "unknown",
                            call_id=tool_call_id,
                            state=tool_state,
                        )
                        tool_parts[tool_call_id] = tool_part
                        assistant_msg_with_parts.parts.append(tool_part)
                        await state.broadcast_event(PartUpdatedEvent.create(tool_part))

                # Tool call complete
                case ToolCallCompleteEvent(
                    tool_call_id=tool_call_id,
                    tool_result=result,
                    metadata=event_metadata,
                ) if tool_call_id in tool_parts:
                    existing = tool_parts[tool_call_id]
                    result_str = str(result) if result else ""
                    tool_input = tool_inputs.get(tool_call_id, {})
                    is_error = isinstance(result, dict) and result.get("error")

                    if is_error:
                        new_state: ToolStateCompleted | ToolStateError = ToolStateError(
                            status="error",
                            error=str(result.get("error", "Unknown error")),
                            input=tool_input,
                            time=TimeStartEnd(start=now, end=now_ms()),
                        )
                    else:
                        new_state = ToolStateCompleted(
                            status="completed",
                            title=f"Completed {existing.tool}",
                            input=tool_input,
                            output=result_str,
                            metadata=event_metadata or {},
                            time=TimeStartEndCompacted(start=now, end=now_ms()),
                        )

                    updated = ToolPart(
                        id=existing.id,
                        message_id=existing.message_id,
                        session_id=existing.session_id,
                        tool=existing.tool,
                        call_id=existing.call_id,
                        state=new_state,
                    )
                    tool_parts[tool_call_id] = updated
                    for i, p in enumerate(assistant_msg_with_parts.parts):
                        if isinstance(p, ToolPart) and p.id == existing.id:
                            assistant_msg_with_parts.parts[i] = updated
                            break
                    await state.broadcast_event(PartUpdatedEvent.create(updated))

                # Stream complete - extract token usage and cost
                case StreamCompleteEvent(message=msg) if msg:
                    if msg.usage:
                        input_tokens = msg.usage.input_tokens or 0
                        output_tokens = msg.usage.output_tokens or 0
                    if msg.cost_info and msg.cost_info.total_cost:
                        # Cost is in Decimal dollars, OpenCode expects float dollars
                        total_cost = float(msg.cost_info.total_cost)

                # Sub-agent/team event - show final results only
                case SubAgentEvent(
                    source_name=source_name,
                    source_type=source_type,
                    event=wrapped_event,
                    depth=depth,
                ):
                    indent = "  " * (depth - 1)

                    match wrapped_event:
                        # Final message from sub-agent/team
                        case StreamCompleteEvent(message=msg):
                            # Show indicator
                            icon = "⚡" if source_type == "team_parallel" else "→"
                            type_label = (
                                " (parallel)"
                                if source_type == "team_parallel"
                                else " (sequential)"
                                if source_type == "team_sequential"
                                else ""
                            )
                            indicator = f"{indent}{icon} {source_name}{type_label}"

                            indicator_part = TextPart(
                                id=identifier.ascending("part"),
                                message_id=assistant_msg_id,
                                session_id=session_id,
                                text=indicator,
                                time=TimeStartEndOptional(start=now_ms()),
                            )
                            assistant_msg_with_parts.parts.append(indicator_part)
                            await state.broadcast_event(PartUpdatedEvent.create(indicator_part))

                            # Show complete message content
                            content = str(msg.content) if msg.content else "(no output)"
                            content_part = TextPart(
                                id=identifier.ascending("part"),
                                message_id=assistant_msg_id,
                                session_id=session_id,
                                text=content,
                                time=TimeStartEndOptional(start=now_ms()),
                            )
                            assistant_msg_with_parts.parts.append(content_part)
                            await state.broadcast_event(PartUpdatedEvent.create(content_part))

                        # Tool call completed - show one-line summary
                        case ToolCallCompleteEvent(tool_name=tool_name, tool_result=result):
                            # Preview result (first 60 chars)
                            result_str = str(result) if result else ""
                            preview = (
                                result_str[:60] + "..." if len(result_str) > 60 else result_str  # noqa: PLR2004
                            )
                            summary = f"{indent}  ├─ {tool_name}: {preview}"

                            summary_part = TextPart(
                                id=identifier.ascending("part"),
                                message_id=assistant_msg_id,
                                session_id=session_id,
                                text=summary,
                                time=TimeStartEndOptional(start=now_ms()),
                            )
                            assistant_msg_with_parts.parts.append(summary_part)
                            await state.broadcast_event(PartUpdatedEvent.create(summary_part))

                # Compaction event - emit session.compacted SSE event
                case CompactionEvent(session_id=compact_session_id, phase=phase):
                    if phase == "completed":
                        await state.broadcast_event(
                            SessionCompactedEvent.create(session_id=compact_session_id)
                        )

    except Exception as e:  # noqa: BLE001
        response_text = f"Error calling agent: {e}"
        # Emit session error event
        await state.broadcast_event(
            SessionErrorEvent.create(
                session_id=session_id,
                error_name=type(e).__name__,
                error_message=str(e),
            )
        )

    response_time = now_ms()

    # Create text part with response (only if we didn't stream it already)
    if response_text and text_part is None:
        text_part = TextPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
            text=response_text,
            time=TimeStartEndOptional(start=now, end=response_time),
        )
        assistant_msg_with_parts.parts.append(text_part)

        # Broadcast text part update
        await state.broadcast_event(PartUpdatedEvent.create(text_part))
    elif text_part is not None:
        # Update the streamed text part with final timing
        final_text_part = TextPart(
            id=text_part.id,
            message_id=assistant_msg_id,
            session_id=session_id,
            text=response_text,
            time=TimeStartEndOptional(start=now, end=response_time),
        )
        # Update in parts list
        for i, p in enumerate(assistant_msg_with_parts.parts):
            if isinstance(p, TextPart) and p.id == text_part.id:
                assistant_msg_with_parts.parts[i] = final_text_part
                break

    step_finish = StepFinishPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        tokens=StepFinishTokens(
            cache=TokenCache(read=0, write=0),
            input=input_tokens,
            output=output_tokens,
            reasoning=0,
        ),
        cost=total_cost,
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(PartUpdatedEvent.create(step_finish))

    print(f"Response text: {response_text[:100] if response_text else 'EMPTY'}...")

    # Update assistant message with final timing and tokens
    updated_assistant = assistant_message.model_copy(
        update={
            "time": MessageTime(created=now, completed=response_time),
            "tokens": Tokens(
                cache=TokensCache(read=0, write=0),
                input=input_tokens,
                output=output_tokens,
                reasoning=0,
            ),
            "cost": total_cost,
        }
    )
    assistant_msg_with_parts.info = updated_assistant

    # Broadcast final message update
    await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
    # Persist assistant message to storage
    await persist_message_to_storage(state, assistant_msg_with_parts, session_id)
    # Mark session as not running
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    await state.broadcast_event(SessionIdleEvent.create(session_id))

    # Update session timestamp
    session = state.sessions[session_id]
    state.sessions[session_id] = session.model_copy(
        update={"time": TimeCreatedUpdated(created=session.time.created, updated=response_time)}
    )
    # Title generation now handled by StorageManager signal (on_title_generated in server.py)
    # Agent calls log_conversation() → _generate_title_from_prompt() → emits title_generated signal
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
async def send_message_async(
    session_id: str,
    request: MessageRequest,
    state: StateDep,
) -> None:
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
