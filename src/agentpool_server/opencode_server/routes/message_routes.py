"""Message routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic_ai import FunctionToolCallEvent
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart as PydanticTextPart,
    TextPartDelta,
    ToolCallPart as PydanticToolCallPart,
)

from agentpool.agents.claude_code_agent.converters import derive_rich_tool_info
from agentpool.agents.events import (
    FileContentItem,
    StreamCompleteEvent,
    TextContentItem,
    ToolCallCompleteEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from agentpool.utils import identifiers as identifier
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict
from agentpool_server.opencode_server.converters import extract_user_prompt_from_parts
from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (  # noqa: TC001
    AssistantMessage,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageUpdatedEvent,
    MessageUpdatedEventProperties,
    MessageWithParts,
    Part,
    PartUpdatedEvent,
    PartUpdatedEventProperties,
    SessionStatus,
    SessionStatusEvent,
    SessionStatusProperties,
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
    from agentpool_server.opencode_server.state import ServerState


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

    from agentpool_server.opencode_server.converters import opencode_to_chat_message

    try:
        # Convert to ChatMessage
        chat_msg = opencode_to_chat_message(msg, conversation_id=session_id)

        # Persist via storage manager
        await state.pool.storage.log_message(chat_msg)
    except Exception:  # noqa: BLE001
        # Don't fail the request if storage fails
        pass


async def _generate_session_title(
    state: ServerState,
    session_id: str,
    user_prompt: str,
    assistant_response: str,
) -> None:
    """Generate a title for the session in the background."""
    from agentpool.messaging.messages import ChatMessage
    from agentpool_server.opencode_server.models import (
        SessionInfoProperties,
        SessionUpdatedEvent,
    )

    try:
        if not state.pool.storage:
            return

        # Create ChatMessage objects for the title generator
        messages = [
            ChatMessage[str](role="user", content=user_prompt),
            ChatMessage[str](role="assistant", content=assistant_response),
        ]

        # Generate title using storage manager
        title = await state.pool.storage.generate_conversation_title(
            messages=messages,
            conversation_id=session_id,
        )

        if title and session_id in state.sessions:
            # Update session with new title
            session = state.sessions[session_id]
            updated_session = session.model_copy(update={"title": title})
            state.sessions[session_id] = updated_session

            # Persist to storage
            from agentpool_server.opencode_server.routes.session_routes import (
                opencode_to_session_data,
            )

            session_data = opencode_to_session_data(
                updated_session,
                agent_name=state.agent.name,
                pool_id=state.pool.manifest.config_file_path,
            )
            await state.pool.sessions.store.save(session_data)

            # Broadcast session update
            await state.broadcast_event(
                SessionUpdatedEvent(properties=SessionInfoProperties(info=updated_session))
            )
    except Exception:  # noqa: BLE001
        # Don't fail if title generation fails
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


@router.post("/message")
async def send_message(  # noqa: PLR0915
    session_id: str,
    request: MessageRequest,
    state: StateDep,
) -> MessageWithParts:
    """Send a message and get response from the agent."""
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
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=user_message))
    )

    # Broadcast user message parts so they appear in UI
    for part in user_parts:
        await state.broadcast_event(
            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=part))
        )

    state.session_status[session_id] = SessionStatus(type="busy")
    props = SessionStatusProperties(session_id=session_id, status=SessionStatus(type="busy"))
    await state.broadcast_event(SessionStatusEvent(properties=props))

    # Extract user prompt text
    user_prompt = extract_user_prompt_from_parts([p.model_dump() for p in request.parts])

    # Create assistant message with sortable ID (must come after user message)
    assistant_msg_id = identifier.ascending("message")
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
        tokens=Tokens(
            cache=TokensCache(read=0, write=0),
            input=0,
            output=0,
            reasoning=0,
        ),
        cost=0,
    )

    # Initialize assistant message with empty parts
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)

    # Broadcast assistant message created
    updated_props = MessageUpdatedEventProperties(info=assistant_message)
    await state.broadcast_event(MessageUpdatedEvent(properties=updated_props))

    # Add step-start part
    step_start = StepStartPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
    )
    assistant_msg_with_parts.parts.append(step_start)
    part_updated_props = PartUpdatedEventProperties(part=step_start)
    await state.broadcast_event(PartUpdatedEvent(properties=part_updated_props))

    # Call the agent
    response_text = ""
    input_tokens = 0
    output_tokens = 0
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
        async for event in agent.run_stream(user_prompt):
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
                    part_updated_props = PartUpdatedEventProperties(part=text_part, delta=delta)
                    await state.broadcast_event(PartUpdatedEvent(properties=part_updated_props))

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
                        part_updated_props = PartUpdatedEventProperties(part=text_part, delta=delta)
                        await state.broadcast_event(PartUpdatedEvent(properties=part_updated_props))

                # Tool call start - from Claude Code agent or toolsets
                case ToolCallStartEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    raw_input=raw_input,
                    title=title,
                ):
                    if tool_call_id in tool_parts:
                        # Update existing part with the custom title
                        existing = tool_parts[tool_call_id]
                        tool_inputs[tool_call_id] = raw_input or tool_inputs.get(tool_call_id, {})

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
                        await state.broadcast_event(
                            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=updated))
                        )
                    else:
                        # Create new tool part with the title
                        tool_inputs[tool_call_id] = raw_input or {}
                        tool_outputs[tool_call_id] = ""
                        tool_state = ToolStateRunning(
                            status="running",
                            time=TimeStart(start=now_ms()),
                            input=raw_input or {},
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
                        part_updated_props = PartUpdatedEventProperties(part=tool_part)
                        await state.broadcast_event(PartUpdatedEvent(properties=part_updated_props))

                # Pydantic-ai tool call events (fallback for pydantic-ai agents)
                case (
                    FunctionToolCallEvent(part=tc_part)
                    | PartStartEvent(part=PydanticToolCallPart() as tc_part)
                ) if tc_part.tool_call_id not in tool_parts:
                    tool_call_id = tc_part.tool_call_id
                    tool_name = tc_part.tool_name
                    raw_input = safe_args_as_dict(tc_part)

                    # Store input and initialize output accumulator
                    tool_inputs[tool_call_id] = raw_input
                    tool_outputs[tool_call_id] = ""

                    # Derive initial title; toolset events may update it later
                    rich_info = derive_rich_tool_info(tool_name, raw_input)
                    tool_state = ToolStateRunning(
                        status="running",
                        time=TimeStart(start=now_ms()),
                        input=raw_input,
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
                    await state.broadcast_event(
                        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=tool_part))
                    )

                # Tool call progress
                case ToolCallProgressEvent(
                    tool_call_id=tool_call_id,
                    title=title,
                    items=items,
                    tool_name=tool_name,
                    tool_input=event_tool_input,
                ) if tool_call_id:
                    # Extract text content from items and accumulate
                    new_output = ""
                    for item in items:
                        if isinstance(item, TextContentItem):
                            new_output += item.text
                        elif isinstance(item, FileContentItem):
                            new_output += item.content

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
                        await state.broadcast_event(
                            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=updated))
                        )
                    else:
                        # Create new tool part from progress event
                        tool_inputs[tool_call_id] = event_tool_input or {}
                        accumulated_output = tool_outputs.get(tool_call_id, "")
                        tool_state = ToolStateRunning(
                            status="running",
                            time=TimeStart(start=now_ms()),
                            input=event_tool_input or {},
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
                        await state.broadcast_event(
                            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=tool_part))
                        )

                # Tool call complete
                case ToolCallCompleteEvent(
                    tool_call_id=tool_call_id,
                    tool_result=result,
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
                    await state.broadcast_event(
                        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=updated))
                    )

                # Stream complete - extract token usage
                case StreamCompleteEvent(message=msg) if msg and msg.usage:
                    input_tokens = msg.usage.input_tokens or 0
                    output_tokens = msg.usage.output_tokens or 0

    except Exception as e:  # noqa: BLE001
        response_text = f"Error calling agent: {e}"

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
        await state.broadcast_event(
            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=text_part))
        )
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
        cost=0,  # TODO: Calculate actual cost
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(
        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=step_finish))
    )

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
        }
    )
    assistant_msg_with_parts.info = updated_assistant

    # Broadcast final message update
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=updated_assistant))
    )

    # Persist assistant message to storage
    await persist_message_to_storage(state, assistant_msg_with_parts, session_id)

    # Mark session as not running
    from agentpool_server.opencode_server.models import SessionIdleEvent, SessionIdleProperties

    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(
        SessionStatusEvent(
            properties=SessionStatusProperties(
                session_id=session_id,
                status=SessionStatus(type="idle"),
            )
        )
    )
    # Also emit deprecated session.idle event (still used by TUI run command)
    await state.broadcast_event(
        SessionIdleEvent(properties=SessionIdleProperties(session_id=session_id))
    )

    # Update session timestamp
    session = state.sessions[session_id]
    state.sessions[session_id] = session.model_copy(
        update={"time": TimeCreatedUpdated(created=session.time.created, updated=response_time)}
    )
    # Trigger title generation if session has default title
    if session.title == "New Session" and state.pool.storage:
        # Convert user_prompt to string if it's a list
        prompt_str = user_prompt if isinstance(user_prompt, str) else str(user_prompt)
        state.create_background_task(
            _generate_session_title(state, session_id, prompt_str, response_text),
            name=f"generate_title_{session_id}",
        )
    return assistant_msg_with_parts


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
