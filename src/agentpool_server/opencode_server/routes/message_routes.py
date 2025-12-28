"""Message routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from agentpool_server.opencode_server import identifier
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
    PartUpdatedEvent,
    PartUpdatedEventProperties,
    SessionStatus,
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
    ToolStatePending,
    ToolStateRunning,
    UserMessage,
)
from agentpool_server.opencode_server.time_utils import now_ms


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
async def send_message(  # noqa: PLR0915
    session_id: str,
    request: MessageRequest,
    state: StateDep,
) -> MessageWithParts:
    """Send a message and get response from the agent."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    now = now_ms()

    # Create user message with sortable ID
    user_msg_id = identifier.ascending("message", request.message_id)

    # Import UserMessageModel
    from agentpool_server.opencode_server.models.message import UserMessageModel

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
    user_parts = [
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

    # Broadcast user message created event
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=user_message))
    )

    # Mark session as running
    from agentpool_server.opencode_server.models import SessionStatusEvent, SessionStatusProperties

    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(
        SessionStatusEvent(
            properties=SessionStatusProperties(
                session_id=session_id,
                status=SessionStatus(type="busy"),
            )
        )
    )

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
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=assistant_message))
    )

    # Add step-start part
    step_start = StepStartPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
    )
    assistant_msg_with_parts.parts.append(step_start)
    await state.broadcast_event(
        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=step_start))
    )

    # Call the agent
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    tool_parts: dict[str, ToolPart] = {}  # Track tool parts by call_id

    if state.agent is not None:
        try:
            # Import event types here to avoid circular imports
            from pydantic_ai.messages import TextPart as PydanticTextPart

            from agentpool.agents.events import (
                StreamCompleteEvent,
                ToolCallCompleteEvent,
                ToolCallProgressEvent,
                ToolCallStartEvent,
            )

            # Stream events from the agent
            async for event in state.agent.run_stream(user_prompt):
                # Handle different event types
                if isinstance(event, ToolCallStartEvent):
                    # Create pending tool part
                    tool_part = ToolPart(
                        id=identifier.ascending("part"),
                        message_id=assistant_msg_id,
                        session_id=session_id,
                        tool=event.tool_name,
                        call_id=event.tool_call_id,
                        state=ToolStatePending(status="pending"),
                    )
                    tool_parts[event.tool_call_id] = tool_part
                    assistant_msg_with_parts.parts.append(tool_part)
                    await state.broadcast_event(
                        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=tool_part))
                    )

                elif isinstance(event, ToolCallProgressEvent):
                    # Update to running state
                    if event.tool_call_id and event.tool_call_id in tool_parts:
                        existing = tool_parts[event.tool_call_id]
                        # Build output from content items
                        output_text = ""
                        for item in event.items:
                            if hasattr(item, "text"):
                                output_text += item.text
                            elif hasattr(item, "content"):
                                output_text += item.content

                        # Extract from existing state safely
                        existing_title = getattr(existing.state, "title", "")
                        existing_input = getattr(existing.state, "input", {})

                        from agentpool_server.opencode_server.models.parts import TimeStart

                        updated = ToolPart(
                            id=existing.id,
                            message_id=existing.message_id,
                            session_id=existing.session_id,
                            tool=existing.tool,
                            call_id=existing.call_id,
                            state=ToolStateRunning(
                                status="running",
                                time=TimeStart(start=now_ms()),
                                title=event.title or existing_title,
                                input=existing_input,
                            ),
                        )
                        tool_parts[event.tool_call_id] = updated
                        # Update in parts list
                        for i, p in enumerate(assistant_msg_with_parts.parts):
                            if isinstance(p, ToolPart) and p.id == existing.id:
                                assistant_msg_with_parts.parts[i] = updated
                                break
                        await state.broadcast_event(
                            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=updated))
                        )

                elif isinstance(event, ToolCallCompleteEvent):
                    # Update to completed/error state
                    if event.tool_call_id in tool_parts:
                        existing = tool_parts[event.tool_call_id]
                        result = event.tool_result
                        result_str = str(result) if result else ""

                        # Extract from existing state safely
                        existing_input = getattr(existing.state, "input", {})

                        # Check result for error indication
                        is_error = isinstance(result, dict) and result.get("error")

                        from agentpool_server.opencode_server.models.parts import (
                            TimeStartEndCompacted,
                        )

                        if is_error:
                            new_state: ToolStateCompleted | ToolStateError = ToolStateError(
                                status="error",
                                error=str(result.get("error", "Unknown error")),
                                input=existing_input,
                                time=TimeStartEnd(start=now, end=now_ms()),
                            )
                        else:
                            new_state = ToolStateCompleted(
                                status="completed",
                                title=f"Completed {existing.tool}",
                                input=existing_input,
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
                        tool_parts[event.tool_call_id] = updated
                        # Update in parts list
                        for i, p in enumerate(assistant_msg_with_parts.parts):
                            if isinstance(p, ToolPart) and p.id == existing.id:
                                assistant_msg_with_parts.parts[i] = updated
                                break
                        await state.broadcast_event(
                            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=updated))
                        )

                elif isinstance(event, StreamCompleteEvent):
                    # Extract final response from ChatMessage
                    if event.message and hasattr(event.message, "parts"):
                        for msg_part in event.message.parts:
                            if isinstance(msg_part, PydanticTextPart):
                                response_text += msg_part.content

                        # Get token usage
                        if hasattr(event.message, "usage") and event.message.usage:
                            input_tokens = event.message.usage.input_tokens or 0
                            output_tokens = event.message.usage.output_tokens or 0

        except Exception as e:  # noqa: BLE001
            response_text = f"Error calling agent: {e}"
    else:
        response_text = (
            "No agent configured. Please start the server with an agent.\n\n"
            "For now, this is a placeholder response. Your message was:\n\n"
            f"> {user_prompt}"
        )

    response_time = now_ms()

    # Create text part with response
    if response_text:
        text_part = TextPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
            text=response_text,
            time=TimeStartEnd(start=now, end=response_time),
        )
        assistant_msg_with_parts.parts.append(text_part)

        # Broadcast text part update
        await state.broadcast_event(
            PartUpdatedEvent(properties=PartUpdatedEventProperties(part=text_part))
        )

    # Add step-finish part with token counts
    from agentpool_server.opencode_server.models.parts import StepFinishTokens, TokenCache

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
