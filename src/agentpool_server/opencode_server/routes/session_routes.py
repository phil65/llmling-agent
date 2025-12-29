"""Session routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

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
    SessionRevert,
    SessionShare,
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
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
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

    # Create input provider for this session
    from agentpool_server.opencode_server.input_provider import OpenCodeInputProvider

    input_provider = OpenCodeInputProvider(state, session_id)
    state.input_providers[session_id] = input_provider

    # Set input provider on agent if configured
    if state.agent is not None:
        state.agent._input_provider = input_provider

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

    # Cancel any pending permissions and clean up input provider
    input_provider = state.input_providers.pop(session_id, None)
    if input_provider is not None:
        input_provider.cancel_all_pending()

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
    """Get todos for a session.

    Returns todos from the agent pool's TodoTracker if available,
    otherwise falls back to session-level todos.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Try to get todos from pool's TodoTracker
    if state.agent is not None and state.agent.agent_pool is not None:
        tracker = state.agent.agent_pool.todos
        return [Todo(id=e.id, content=e.content, status=e.status) for e in tracker.entries]

    # Fall back to session-level todos (legacy)
    return state.todos.get(session_id, [])


@router.get("/{session_id}/diff")
async def get_session_diff(
    session_id: str,
    state: StateDep,
    message_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get file diffs for a session.

    Returns a list of file changes with unified diffs.
    Optionally filter to changes since a specific message.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get file operations from the agent's pool
    if state.agent is None or state.agent.agent_pool is None:
        return []

    file_ops = state.agent.agent_pool.file_ops
    if not file_ops.changes:
        return []

    # Optionally filter by message_id
    changes = file_ops.get_changes_since_message(message_id) if message_id else file_ops.changes

    # Format as list of diffs
    return [
        {
            "path": change.path,
            "operation": change.operation,
            "diff": change.to_unified_diff(),
            "timestamp": change.timestamp,
            "agent_name": change.agent_name,
            "message_id": change.message_id,
        }
        for change in changes
    ]


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


class PermissionResponse(OpenCodeBaseModel):
    """Request body for responding to a permission request."""

    response: Literal["once", "always", "reject"]


@router.get("/{session_id}/permissions")
async def get_pending_permissions(session_id: str, state: StateDep) -> list[dict[str, Any]]:
    """Get all pending permission requests for a session.

    Returns a list of pending permissions awaiting user response.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the input provider for this session
    input_provider = state.input_providers.get(session_id)
    if input_provider is None:
        return []

    return input_provider.get_pending_permissions()


@router.post("/{session_id}/permissions/{permission_id}")
async def respond_to_permission(
    session_id: str,
    permission_id: str,
    request: PermissionResponse,
    state: StateDep,
) -> bool:
    """Respond to a pending permission request.

    The response can be:
    - "once": Allow this tool execution once
    - "always": Always allow this tool (remembered for session)
    - "reject": Reject this tool execution
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the input provider for this session
    input_provider = state.input_providers.get(session_id)
    if input_provider is None:
        raise HTTPException(status_code=404, detail="No input provider for session")

    # Resolve the permission
    resolved = input_provider.resolve_permission(permission_id, request.response)
    if not resolved:
        raise HTTPException(status_code=404, detail="Permission not found or already resolved")

    # Broadcast the resolution event
    from agentpool_server.opencode_server.models.events import (
        PermissionResolvedEvent,
        PermissionResolvedProperties,
    )

    await state.broadcast_event(
        PermissionResolvedEvent(
            properties=PermissionResolvedProperties(
                session_id=session_id,
                permission_id=permission_id,
                response=request.response,
            )
        )
    )

    return True


@router.post("/{session_id}/summarize")
async def summarize_session(session_id: str, state: StateDep) -> MessageWithParts:
    """Summarize the session conversation.

    Uses the Summarize compaction step to condense older messages
    into a summary while keeping recent messages intact.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.agent is None:
        raise HTTPException(status_code=400, detail="No agent configured")

    messages = state.messages.get(session_id, [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to summarize")

    now = now_ms()

    # Create assistant message for the summary
    assistant_msg_id = identifier.ascending("message")
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id="",
        model_id="summarizer",
        provider_id="agentpool",
        mode="summarize",
        agent="summarizer",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now, completed=None),
        tokens=Tokens(cache=TokensCache(read=0, write=0), input=0, output=0, reasoning=0),
        cost=0,
    )

    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])

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

    try:
        from agentpool.messaging.compaction import compact_conversation, summarizing_context

        # Get the compaction pipeline from the agent pool configuration
        pipeline = None
        if state.agent.agent_pool is not None:
            pipeline = state.agent.agent_pool.compaction_pipeline

        if pipeline is None:
            # Fall back to a default summarizing pipeline
            pipeline = summarizing_context()

        # Apply the compaction pipeline using shared helper
        original_count, compacted_count = await compact_conversation(
            pipeline, state.agent.conversation
        )

        if original_count > 0:
            output_text = (
                f"Conversation compacted using configured pipeline.\n"
                f"Messages reduced from {original_count} to {compacted_count}."
            )
        else:
            output_text = "No conversation history to compact."

    except Exception as e:  # noqa: BLE001
        output_text = f"Error summarizing session: {e}"

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

    # Add the summary message to the session
    state.messages[session_id].append(assistant_msg_with_parts)

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


@router.post("/{session_id}/share")
async def share_session(
    session_id: str,
    state: StateDep,
    num_messages: int | None = None,
) -> SessionShare:
    """Share session conversation history via OpenCode's sharing service.

    Uses the OpenCode share API to create a shareable link with the full
    conversation including messages and parts.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = state.sessions[session_id]
    messages = state.messages.get(session_id, [])

    if not messages:
        raise HTTPException(status_code=400, detail="No messages to share")

    # Apply message limit if specified
    if num_messages is not None and num_messages > 0:
        messages = messages[-num_messages:]

    from anyenv.text_sharing.opencode import Message, MessagePart, OpenCodeSharer

    # Convert our messages to OpenCode Message format
    opencode_messages: list[Message] = []

    for msg_with_parts in messages:
        info = msg_with_parts.info
        role = getattr(info, "role", "user")

        # Map role to OpenCode roles
        if role not in ("user", "assistant", "system"):
            role = "assistant" if role == "model" else "user"

        # Extract text parts
        parts: list[MessagePart] = []
        for part in msg_with_parts.parts:
            if hasattr(part, "text") and part.text:
                parts.append(MessagePart(type="text", text=part.text))

        if parts:
            opencode_messages.append(Message(role=role, parts=parts))

    if not opencode_messages:
        raise HTTPException(status_code=400, detail="No content to share")

    # Share via OpenCode API
    async with OpenCodeSharer() as sharer:
        result = await sharer.share_conversation(
            opencode_messages,
            title=session.title,
        )
        share_url = result.url

    # Store the share URL in the session
    share_info = SessionShare(url=share_url)
    updated_session = session.model_copy(update={"share": share_info})
    state.sessions[session_id] = updated_session

    # Broadcast session update
    await state.broadcast_event(
        SessionUpdatedEvent(properties=SessionInfoProperties(info=updated_session))
    )

    return share_info


class RevertRequest(OpenCodeBaseModel):
    """Request body for reverting a message."""

    message_id: str = Field(alias="messageID")
    part_id: str | None = Field(default=None, alias="partID")


@router.post("/{session_id}/revert")
async def revert_session(
    session_id: str,
    request: RevertRequest,
    state: StateDep,
) -> Session:
    """Revert file changes from a specific message.

    Restores files to their state before the specified message's changes.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.agent is None or state.agent.agent_pool is None:
        raise HTTPException(status_code=400, detail="No agent configured")

    file_ops = state.agent.agent_pool.file_ops
    if not file_ops.changes:
        raise HTTPException(status_code=400, detail="No file changes to revert")

    # Get revert operations for changes since this message
    revert_ops = file_ops.get_revert_operations(since_message_id=request.message_id)

    if not revert_ops:
        raise HTTPException(
            status_code=404,
            detail=f"No changes found for message {request.message_id}",
        )

    # Get filesystem from the agent's environment
    fs = state.agent.env.get_fs()

    # Apply reverts using the filesystem
    # TODO: Currently write operations only track "existed vs created", not full old content.
    # Files that existed before a write will be restored as empty, not their original content.
    reverted_paths = []
    for path, content in revert_ops:
        try:
            if content is None:
                # File was created (old_text=None), delete it
                await fs._rm_file(path)
            else:
                # Restore original content
                content_bytes = content.encode("utf-8")
                await fs._pipe_file(path, content_bytes)
            reverted_paths.append(path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to revert {path}: {e}",
            ) from e

    # Remove the reverted changes from the tracker
    file_ops.remove_changes_since_message(request.message_id)

    # Update session with revert info
    session = state.sessions[session_id]
    revert_info = SessionRevert(
        message_id=request.message_id,
        diff=None,  # Could include the revert diff here
        part_id=request.part_id,
    )
    updated_session = session.model_copy(update={"revert": revert_info})
    state.sessions[session_id] = updated_session

    # Broadcast session update
    await state.broadcast_event(
        SessionUpdatedEvent(properties=SessionInfoProperties(info=updated_session))
    )

    return updated_session


@router.post("/{session_id}/unrevert")
async def unrevert_session(session_id: str, state: StateDep) -> Session:
    """Restore all reverted file changes.

    Re-applies the changes that were previously reverted.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.agent is None or state.agent.agent_pool is None:
        raise HTTPException(status_code=400, detail="No agent configured")

    file_ops = state.agent.agent_pool.file_ops
    if not file_ops.reverted_changes:
        raise HTTPException(status_code=400, detail="No reverted changes to restore")

    # Get unrevert operations
    unrevert_ops = file_ops.get_unrevert_operations()

    # Get filesystem from the agent's environment
    fs = state.agent.env.get_fs()

    # Apply unrevert - write back the new_content
    for path, content in unrevert_ops:
        try:
            if content is None:
                # File was deleted in the original change, delete it again
                await fs._rm_file(path)
            else:
                # Restore the changed content
                content_bytes = content.encode("utf-8")
                await fs._pipe_file(path, content_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to unrevert {path}: {e}",
            ) from e

    # Restore the changes to the tracker
    file_ops.restore_reverted_changes()

    # Clear revert info from session
    session = state.sessions[session_id]
    updated_session = session.model_copy(update={"revert": None})
    state.sessions[session_id] = updated_session

    # Broadcast session update
    await state.broadcast_event(
        SessionUpdatedEvent(properties=SessionInfoProperties(info=updated_session))
    )

    return updated_session


@router.delete("/{session_id}/share")
async def unshare_session(session_id: str, state: StateDep) -> bool:
    """Remove share link from a session.

    Note: This only removes the link from the session metadata.
    The shared content may still exist on the provider's servers.
    """
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = state.sessions[session_id]
    if session.share is None:
        raise HTTPException(status_code=400, detail="Session is not shared")

    # Remove share info from session
    updated_session = session.model_copy(update={"share": None})
    state.sessions[session_id] = updated_session

    # Broadcast session update
    await state.broadcast_event(
        SessionUpdatedEvent(properties=SessionInfoProperties(info=updated_session))
    )

    return True


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
