"""Session routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentpool_server.opencode_server import identifier
from agentpool_server.opencode_server.dependencies import StateDep  # noqa: TC001
from agentpool_server.opencode_server.models import (  # noqa: TC001
    AssistantMessage,
    CommandRequest,
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageUpdatedEventProperties,
    MessageWithParts,
    PartUpdatedEvent,
    PartUpdatedEventProperties,
    Session,
    SessionCreatedEvent,
    SessionCreateRequest,
    SessionDeletedEvent,
    SessionDeletedProperties,
    SessionInfoProperties,
    SessionStatus,
    SessionStatusEvent,
    SessionStatusProperties,
    SessionUpdatedEvent,
    SessionUpdateRequest,
    ShellRequest,
    StepFinishPart,
    StepStartPart,
    TextPart,
    TimeCreatedUpdated,
    Todo,
    Tokens,
    TokensCache,
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


@router.post("/{session_id}/shell")
async def run_shell_command(
    session_id: str,
    request: ShellRequest,
    state: StateDep,
) -> MessageWithParts:
    """Run a shell command directly."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    now = now_ms()

    # Create assistant message for the shell output
    assistant_msg_id = identifier.ascending("message")
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id="",  # Shell commands don't have a parent user message
        model_id=request.model.model_id if request.model else "shell",
        provider_id=request.model.provider_id if request.model else "local",
        mode="shell",
        agent=request.agent,
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now, completed=None),
        tokens=Tokens(cache=TokensCache(read=0, write=0), input=0, output=0, reasoning=0),
        cost=0,
    )

    # Initialize message with empty parts
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)

    # Broadcast message created
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=assistant_message))
    )

    # Mark session as busy
    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(
        SessionStatusEvent(
            properties=SessionStatusProperties(
                session_id=session_id,
                status=SessionStatus(type="busy"),
            )
        )
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

    # Execute the command
    output_text = ""
    success = False

    if state.agent is not None and hasattr(state.agent, "env"):
        try:
            result = await state.agent.env.execute_command(request.command)
            success = result.success
            if success:
                output_text = str(result.result) if result.result else ""
            else:
                output_text = f"Error: {result.error}" if result.error else "Command failed"
        except Exception as e:  # noqa: BLE001
            output_text = f"Error executing command: {e}"
    else:
        # Fallback: use subprocess directly
        import asyncio

        try:
            proc = await asyncio.create_subprocess_shell(
                request.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=state.working_dir,
            )
            stdout, stderr = await proc.communicate()
            success = proc.returncode == 0
            output_text = stdout.decode() if success else stderr.decode()
        except Exception as e:  # noqa: BLE001
            output_text = f"Error: {e}"

    response_time = now_ms()

    # Create text part with output
    text_part = TextPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        text=f"$ {request.command}\n{output_text}",
    )
    assistant_msg_with_parts.parts.append(text_part)
    await state.broadcast_event(
        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=text_part))
    )

    # Add step-finish part
    from agentpool_server.opencode_server.models.parts import StepFinishTokens, TokenCache

    step_finish = StepFinishPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        tokens=StepFinishTokens(
            cache=TokenCache(read=0, write=0),
            input=0,
            output=0,
            reasoning=0,
        ),
        cost=0,
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(
        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=step_finish))
    )

    # Update message with completion time
    updated_assistant = assistant_message.model_copy(
        update={"time": MessageTime(created=now, completed=response_time)}
    )
    assistant_msg_with_parts.info = updated_assistant
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=updated_assistant))
    )

    # Mark session as idle
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(
        SessionStatusEvent(
            properties=SessionStatusProperties(
                session_id=session_id,
                status=SessionStatus(type="idle"),
            )
        )
    )

    return assistant_msg_with_parts


@router.post("/{session_id}/command")
async def execute_command(  # noqa: PLR0915
    session_id: str,
    request: CommandRequest,
    state: StateDep,
) -> MessageWithParts:
    """Execute a slash command (MCP prompt).

    Commands are mapped to MCP prompts. The command name is used to find
    the matching prompt, and arguments are parsed and passed to it.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get available prompts from agent
    if state.agent is None or not hasattr(state.agent, "tools"):
        raise HTTPException(status_code=400, detail="No agent configured")

    prompts = await state.agent.tools.list_prompts()

    # Find matching prompt by name
    prompt = next((p for p in prompts if p.name == request.command), None)
    if prompt is None:
        raise HTTPException(
            status_code=404,
            detail=f"Command not found: {request.command}",
        )

    # Parse arguments - OpenCode uses $1, $2 style, MCP uses named arguments
    # For simplicity, we'll pass the raw arguments string to the first argument
    # or parse space-separated args into a dict
    arguments: dict[str, str] = {}
    if request.arguments and prompt.arguments:
        # Split arguments and map to prompt argument names
        arg_values = request.arguments.split()
        for i, arg_def in enumerate(prompt.arguments):
            if i < len(arg_values):
                arguments[arg_def["name"]] = arg_values[i]

    now = now_ms()

    # Create assistant message
    assistant_msg_id = identifier.ascending("message")
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id="",
        model_id=request.model or "default",
        provider_id="mcp",
        mode="command",
        agent=request.agent or "default",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now, completed=None),
        tokens=Tokens(cache=TokensCache(read=0, write=0), input=0, output=0, reasoning=0),
        cost=0,
    )

    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)

    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=assistant_message))
    )

    # Mark session as busy
    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(
        SessionStatusEvent(
            properties=SessionStatusProperties(
                session_id=session_id,
                status=SessionStatus(type="busy"),
            )
        )
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

    # Get prompt content and execute through the agent
    try:
        prompt_parts = await prompt.get_components(arguments)
        # Extract text content from parts
        prompt_texts = []
        for part in prompt_parts:
            if hasattr(part, "content"):
                content = part.content
                if isinstance(content, str):
                    prompt_texts.append(content)
                elif isinstance(content, list):
                    # Handle Sequence[UserContent]
                    for item in content:
                        if hasattr(item, "text"):
                            prompt_texts.append(item.text)
                        elif isinstance(item, str):
                            prompt_texts.append(item)
        prompt_text = "\n".join(prompt_texts)

        # Run the expanded prompt through the agent
        result = await state.agent.run(prompt_text)
        output_text = str(result.output) if hasattr(result, "output") else str(result)

    except Exception as e:  # noqa: BLE001
        output_text = f"Error executing command: {e}"

    response_time = now_ms()

    # Create text part with output
    text_part = TextPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        text=output_text,
    )
    assistant_msg_with_parts.parts.append(text_part)
    await state.broadcast_event(
        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=text_part))
    )

    # Add step-finish part
    from agentpool_server.opencode_server.models.parts import StepFinishTokens, TokenCache

    step_finish = StepFinishPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        tokens=StepFinishTokens(
            cache=TokenCache(read=0, write=0),
            input=0,
            output=0,
            reasoning=0,
        ),
        cost=0,
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(
        PartUpdatedEvent(properties=PartUpdatedEventProperties(part=step_finish))
    )

    # Update message with completion time
    updated_assistant = assistant_message.model_copy(
        update={"time": MessageTime(created=now, completed=response_time)}
    )
    assistant_msg_with_parts.info = updated_assistant
    await state.broadcast_event(
        MessageUpdatedEvent(properties=MessageUpdatedEventProperties(info=updated_assistant))
    )

    # Mark session as idle
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(
        SessionStatusEvent(
            properties=SessionStatusProperties(
                session_id=session_id,
                status=SessionStatus(type="idle"),
            )
        )
    )

    return assistant_msg_with_parts
