"""Session routes."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from anyenv.text_sharing.opencode import Message, MessagePart, OpenCodeSharer
from fastapi import APIRouter, HTTPException
from pydantic_ai import FileUrl

from agentpool.repomap import RepoMap, find_src_files
from agentpool.sessions.models import SessionData
from agentpool.utils import identifiers as identifier
from agentpool_config.session import SessionQuery
from agentpool_server.opencode_server.command_validation import validate_command
from agentpool_server.opencode_server.converters import chat_message_to_opencode
from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.input_provider import OpenCodeInputProvider
from agentpool_server.opencode_server.models import (
    AssistantMessage,
    CommandRequest,
    MessagePath,
    MessageTime,
    MessageUpdatedEvent,
    MessageWithParts,
    PartUpdatedEvent,
    Session,
    SessionCreatedEvent,
    SessionCreateRequest,
    SessionDeletedEvent,
    SessionForkRequest,
    SessionInitRequest,
    SessionRevert,
    SessionShare,
    SessionStatus,
    SessionStatusEvent,
    SessionUpdatedEvent,
    SessionUpdateRequest,
    ShellRequest,
    StepFinishPart,
    StepStartPart,
    SummarizeRequest,
    TextPart,
    TimeCreatedUpdated,
    Todo,
    Tokens,
    TokensCache,
)
from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.events import PermissionResolvedEvent
from agentpool_server.opencode_server.models.parts import StepFinishTokens, TokenCache
from agentpool_server.opencode_server.time_utils import now_ms


if TYPE_CHECKING:
    from agentpool_server.opencode_server.state import ServerState


# =============================================================================
# Conversion helpers between OpenCode Session and SessionData
# =============================================================================


def session_data_to_opencode(
    data: SessionData,
    title: str | None = None,
) -> Session:
    """Convert SessionData to OpenCode Session model.

    Args:
        data: SessionData to convert
        title: Optional title (fetched from storage by caller)
    """
    # Convert datetime to milliseconds timestamp
    created_ms = int(data.created_at.timestamp() * 1000)
    updated_ms = int(data.last_active.timestamp() * 1000)

    # Extract revert/share from metadata if present
    revert = None
    share = None
    if "revert" in data.metadata:
        revert = SessionRevert(**data.metadata["revert"])
    if "share" in data.metadata:
        share = SessionShare(**data.metadata["share"])

    return Session(
        id=data.session_id,
        project_id=data.project_id or "default",
        directory=data.cwd or "",
        title=title or "New Session",
        version=data.version,
        time=TimeCreatedUpdated(created=created_ms, updated=updated_ms),
        parent_id=data.parent_id,
        revert=revert,
        share=share,
    )


def opencode_to_session_data(
    session: Session,
    *,
    agent_name: str = "default",
    pool_id: str | None = None,
) -> SessionData:
    """Convert OpenCode Session to SessionData for persistence."""
    # Convert milliseconds timestamp to datetime
    created_at = datetime.fromtimestamp(session.time.created / 1000, tz=UTC)
    last_active = datetime.fromtimestamp(session.time.updated / 1000, tz=UTC)
    # Store revert/share in metadata
    metadata: dict[str, Any] = {}
    if session.revert:
        metadata["revert"] = session.revert.model_dump()
    if session.share:
        metadata["share"] = session.share.model_dump()
    return SessionData(
        session_id=session.id,
        agent_name=agent_name,
        conversation_id=session.id,  # Use session_id as conversation_id
        pool_id=pool_id,
        project_id=session.project_id,
        parent_id=session.parent_id,
        version=session.version,
        cwd=session.directory,
        created_at=created_at,
        last_active=last_active,
        metadata=metadata,
    )


async def load_messages_from_storage(state: ServerState, session_id: str) -> list[MessageWithParts]:
    """Load messages from storage and convert to OpenCode format.

    Args:
        state: Server state with pool reference
        session_id: Session/conversation ID

    Returns:
        List of OpenCode MessageWithParts
    """
    if state.pool.storage is None:
        return []

    try:
        query = SessionQuery(name=session_id)  # conversation_id = session_id
        chat_messages = await state.pool.storage.filter_messages(query)
        # Convert to OpenCode format
        opencode_messages = []
        working_dir = state.working_dir
        agent_name = state.agent.name
        for chat_msg in chat_messages:
            opencode_msg = chat_message_to_opencode(
                chat_msg,
                session_id=session_id,
                working_dir=working_dir,
                agent_name=agent_name,
                model_id=chat_msg.model_name or "unknown",
                provider_id=chat_msg.provider_name or "agentpool",
            )
            opencode_messages.append(opencode_msg)

    except Exception:  # noqa: BLE001
        # If storage fails, return empty list
        return []
    else:
        return opencode_messages


async def get_or_load_session(state: ServerState, session_id: str) -> Session | None:
    """Get session from cache or load from storage.

    Returns None if session not found in either location.
    Also loads messages from storage if not already cached.
    """
    # Check in-memory cache first
    if session_id in state.sessions:
        return state.sessions[session_id]

    # Try to load from storage
    data = await state.pool.sessions.store.load(session_id)
    if data is not None:
        # Fetch title from conversation storage
        title = None
        if state.pool.storage:
            title = await state.pool.storage.get_conversation_title(data.conversation_id)
        session = session_data_to_opencode(data, title=title)
        # Cache it
        state.sessions[session_id] = session
        # Initialize runtime state
        if session_id not in state.session_status:
            state.session_status[session_id] = SessionStatus(type="idle")
        # Load messages from storage if not cached
        if session_id not in state.messages:
            state.messages[session_id] = await load_messages_from_storage(state, session_id)
        return session

    return None


router = APIRouter(prefix="/session", tags=["session"])


@router.get("")
async def list_sessions(state: StateDep) -> list[Session]:
    """List all sessions from storage.

    Returns all persisted sessions, not just active ones.
    """
    sessions: list[Session] = []
    # Load all session IDs from storage
    session_ids = await state.pool.sessions.store.list_sessions()
    for session_id in session_ids:
        # Use get_or_load to populate cache and get Session model
        session = await get_or_load_session(state, session_id)
        if session is not None:
            sessions.append(session)

    return sessions


@router.post("")
async def create_session(state: StateDep, request: SessionCreateRequest | None = None) -> Session:
    """Create a new session and persist to storage."""
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

    # Persist to storage
    id_ = state.pool.manifest.config_file_path
    session_data = opencode_to_session_data(session, agent_name=state.agent.name, pool_id=id_)
    await state.pool.sessions.store.save(session_data)
    # Cache in memory
    state.sessions[session_id] = session
    state.messages[session_id] = []
    state.session_status[session_id] = SessionStatus(type="idle")
    state.todos[session_id] = []
    # Create input provider for this session
    input_provider = OpenCodeInputProvider(state, session_id)
    state.input_providers[session_id] = input_provider
    # Set input provider on agent
    state.agent._input_provider = input_provider
    await state.broadcast_event(SessionCreatedEvent.create(session))
    return session


@router.get("/status")
async def get_session_status(state: StateDep) -> dict[str, SessionStatus]:
    """Get status for all sessions.

    Returns only non-idle sessions. If all sessions are idle, returns empty dict.
    """
    return {sid: status for sid, status in state.session_status.items() if status.type != "idle"}


@router.get("/{session_id}")
async def get_session(session_id: str, state: StateDep) -> Session:
    """Get session details.

    Loads from storage if not in memory cache.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    state: StateDep,
) -> Session:
    """Update session properties and persist changes."""
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.title is not None:
        time_ = TimeCreatedUpdated(created=session.time.created, updated=now_ms())
        session = session.model_copy(update={"title": request.title, "time": time_})
    state.sessions[session_id] = session  # Update cache
    id_ = state.pool.manifest.config_file_path
    session_data = opencode_to_session_data(session, agent_name=state.agent.name, pool_id=id_)
    await state.pool.sessions.store.save(session_data)
    await state.broadcast_event(SessionUpdatedEvent.create(session))
    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str, state: StateDep) -> bool:
    """Delete a session from both cache and storage."""
    # Check if session exists (in cache or storage)
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel any pending permissions and clean up input provider
    input_provider = state.input_providers.pop(session_id, None)
    if input_provider is not None:
        input_provider.cancel_all_pending()

    # Remove from cache
    state.sessions.pop(session_id, None)
    state.messages.pop(session_id, None)
    state.session_status.pop(session_id, None)
    state.todos.pop(session_id, None)
    # Delete from storage
    await state.pool.sessions.store.delete(session_id)
    await state.broadcast_event(SessionDeletedEvent.create(session_id))

    return True


@router.post("/{session_id}/abort")
async def abort_session(session_id: str, state: StateDep) -> bool:
    """Abort a running session by interrupting the agent."""
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Interrupt the agent to cancel any ongoing stream
    try:
        await state.agent.interrupt()
        # Give a moment for the cancellation to propagate
        await asyncio.sleep(0.1)
    except Exception:  # noqa: BLE001
        pass

    # Update and broadcast session status to notify clients
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))

    return True


@router.post("/{session_id}/fork")
async def fork_session(  # noqa: D417
    session_id: str,
    state: StateDep,
    request: SessionForkRequest | None = None,
    directory: str | None = None,
) -> Session:
    """Fork a session, optionally at a specific message.

    Creates a new session with:
    - parent_id pointing to the original session
    - Copies all messages (or up to message_id if specified)
    - Independent conversation history from that point forward

    Args:
        session_id: The session to fork from
        request: Optional fork parameters (message_id to fork from)
        directory: Optional directory for the forked session

    Returns:
        The newly created forked session
    """
    # Get the original session
    original_session = await get_or_load_session(state, session_id)
    if original_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get messages from the original session
    original_messages = state.messages.get(session_id, [])
    # Filter messages if message_id is specified
    messages_to_copy: list[MessageWithParts] = []
    if request and request.message_id:
        # Copy messages up to and including the specified message_id
        for msg in original_messages:
            messages_to_copy.append(msg)
            if msg.info.id == request.message_id:
                break
        else:
            # message_id not found in messages
            detail = f"Message {request.message_id} not found in session"
            raise HTTPException(status_code=404, detail=detail)
    else:
        # Copy all messages
        messages_to_copy = list(original_messages)

    # Create the new forked session
    now = now_ms()
    new_session_id = identifier.ascending("session")
    # Use provided directory or inherit from original session
    fork_directory = directory if directory else original_session.directory
    forked_session = Session(
        id=new_session_id,
        project_id=original_session.project_id,
        directory=fork_directory,
        title=f"{original_session.title} (fork)",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
        parent_id=session_id,  # Link to original session
    )

    # Persist the forked session to storage
    session_data = opencode_to_session_data(
        forked_session,
        agent_name=state.agent.name,
        pool_id=state.pool.manifest.config_file_path,
    )
    await state.pool.sessions.store.save(session_data)
    # Cache in memory
    state.sessions[new_session_id] = forked_session
    state.session_status[new_session_id] = SessionStatus(type="idle")
    state.todos[new_session_id] = []
    # Copy messages to the new session (with updated session_id references)
    copied_messages: list[MessageWithParts] = []
    for msg_with_parts in messages_to_copy:
        # Create new message info with updated session_id
        new_info = msg_with_parts.info.model_copy(update={"session_id": new_session_id})
        # Copy parts with updated session_id
        new_parts = [
            part.model_copy(update={"session_id": new_session_id}) for part in msg_with_parts.parts
        ]
        copied_messages.append(MessageWithParts(info=new_info, parts=new_parts))

    state.messages[new_session_id] = copied_messages
    input_provider = OpenCodeInputProvider(state, new_session_id)
    state.input_providers[new_session_id] = input_provider
    # Broadcast session created event
    await state.broadcast_event(SessionCreatedEvent.create(forked_session))
    return forked_session


@router.post("/{session_id}/init")
async def init_session(  # noqa: D417
    session_id: str,
    state: StateDep,
    request: SessionInitRequest | None = None,
) -> bool:
    """Initialize a session by analyzing the codebase and creating AGENTS.md.

    Generates a repository map, reads README if present, and runs the agent
    with a prompt to create an AGENTS.md file with project-specific context.

    Args:
        session_id: The session to initialize
        request: Optional model/provider override for the init task

    Returns:
        True when the init task has been started (runs async)
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the agent and filesystem
    agent = state.agent
    fs = agent.env.get_fs()
    working_dir = state.working_dir
    try:
        all_files = await find_src_files(fs, working_dir)
        repo_map = RepoMap(fs=fs, root_path=working_dir, max_tokens=4000)
        repomap_content = await repo_map.get_map(all_files) or "No repository map generated."
    except Exception as e:  # noqa: BLE001
        repomap_content = f"Error generating repository map: {e}"

    # Try to read README.md
    readme_content = ""
    for readme_name in ["README.md", "readme.md", "README", "readme.txt"]:
        try:
            readme_path = f"{working_dir}/{readme_name}".replace("//", "/")
            content = await fs._cat_file(readme_path)
            readme_content = content.decode("utf-8") if isinstance(content, bytes) else content
            break
        except Exception:  # noqa: BLE001
            continue

    # Build the init prompt
    prompt_parts = [
        "Please analyze this codebase and create an AGENTS.md file in the project root.",
        "",
        "<repository-structure>",
        repomap_content,
        "</repository-structure>",
    ]
    if readme_content:
        prompt_parts.extend(["", "<readme>", readme_content, "</readme>"])
    prompt_parts.extend([
        "",
        "Include:",
        "1. Build/lint/test commands - especially for running a single test",
        "2. Code style guidelines (imports, formatting, types, naming conventions, error handling)",
        "",
        "The file will be given to AI coding agents working in this repository. "
        "Keep it around 150 lines.",
        "",
        "If there are existing rules (.cursor/rules/, .cursorrules, "
        ".github/copilot-instructions.md), incorporate them.",
    ])

    init_prompt = "\n".join(prompt_parts)

    # Handle model selection if requested
    original_model: str | None = None
    if request and request.model_id and request.provider_id:
        requested_model = f"{request.provider_id}:{request.model_id}"
        try:
            available_models = await agent.get_available_models()
            if available_models:
                valid_ids = [m.id_override if m.id_override else m.id for m in available_models]
                if requested_model in valid_ids:
                    # Store original model to restore later
                    original_model = agent.model_name
                    await agent.set_model(requested_model)
        except Exception:  # noqa: BLE001
            # Agent doesn't support model selection, ignore
            pass

    # Run the agent in the background
    async def run_init() -> None:
        try:
            await agent.run(init_prompt)
        finally:
            # Restore original model if we changed it
            if original_model is not None:
                with contextlib.suppress(Exception):
                    await agent.set_model(original_model)

    state.create_background_task(run_init(), name=f"init_{session_id}")

    return True


@router.get("/{session_id}/todo")
async def get_session_todos(session_id: str, state: StateDep) -> list[Todo]:
    """Get todos for a session.

    Returns todos from the agent pool's TodoTracker.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get todos from pool's TodoTracker
    tracker = state.pool.todos
    return [Todo(id=e.id, content=e.content, status=e.status) for e in tracker.entries]


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
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    file_ops = state.pool.file_ops
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
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate command for security issues
    validate_command(request.command, state.working_dir)
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
        cost=0.0,
    )

    # Initialize message with empty parts
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)
    # Broadcast message created
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
    # Mark session as busy
    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="busy")))
    # Add step-start part
    step_start = StepStartPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
    )
    assistant_msg_with_parts.parts.append(step_start)
    await state.broadcast_event(PartUpdatedEvent.create(step_start))
    # Execute the command
    output_text = ""
    success = False
    try:
        result = await state.agent.env.execute_command(request.command)
        success = result.success
        if success:
            output_text = str(result.result) if result.result else ""
        else:
            output_text = f"Error: {result.error}" if result.error else "Command failed"
    except Exception as e:  # noqa: BLE001
        output_text = f"Error executing command: {e}"

    response_time = now_ms()
    # Create text part with output
    text_part = TextPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        text=f"$ {request.command}\n{output_text}",
    )
    assistant_msg_with_parts.parts.append(text_part)
    await state.broadcast_event(PartUpdatedEvent.create(text_part))
    step_finish = StepFinishPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        tokens=StepFinishTokens(cache=TokenCache(read=0, write=0)),
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(PartUpdatedEvent.create(step_finish))
    # Update message with completion time
    time_ = MessageTime(created=now, completed=response_time)
    updated_assistant = assistant_message.model_copy(update={"time": time_})
    assistant_msg_with_parts.info = updated_assistant
    await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
    # Mark session as idle
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    return assistant_msg_with_parts


class PermissionResponse(OpenCodeBaseModel):
    """Request body for responding to a permission request."""

    reply: Literal["once", "always", "reject"]


@router.get("/{session_id}/permissions")
async def get_pending_permissions(session_id: str, state: StateDep) -> list[dict[str, Any]]:
    """Get all pending permission requests for a session.

    Returns a list of pending permissions awaiting user response.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
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
    body: PermissionResponse,
    state: StateDep,
) -> bool:
    """Respond to a pending permission request.

    The response can be:
    - "once": Allow this tool execution once
    - "always": Always allow this tool (remembered for session)
    - "reject": Reject this tool execution
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the input provider for this session
    input_provider = state.input_providers.get(session_id)
    if input_provider is None:
        raise HTTPException(status_code=404, detail="No input provider for session")

    # Resolve the permission
    resolved = input_provider.resolve_permission(permission_id, body.reply)
    if not resolved:
        raise HTTPException(status_code=404, detail="Permission not found or already resolved")

    await state.broadcast_event(
        PermissionResolvedEvent.create(
            session_id=session_id,
            request_id=permission_id,
            reply=body.reply,
        )
    )

    return True


# OpenCode-style continuation prompt for summarization
SUMMARIZE_PROMPT = """Provide a detailed prompt for continuing our conversation above. Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next considering new session will not have access to our conversation."""  # noqa: E501


@router.post("/{session_id}/summarize")
async def summarize_session(  # noqa: PLR0915
    session_id: str,
    state: StateDep,
    request: SummarizeRequest | None = None,
) -> MessageWithParts:
    """Summarize the session conversation.

    First runs the compaction pipeline to condense older messages,
    then streams an LLM-generated summary/continuation prompt to the user.
    The summary message is marked with summary=true for UI display.
    """
    from pydantic_ai.messages import (
        PartDeltaEvent,
        PartStartEvent,
        TextPart as PydanticTextPart,
        TextPartDelta,
    )

    from agentpool.agents.events import StreamCompleteEvent

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = state.messages.get(session_id, [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to summarize")

    # Determine model to use
    model_id = request.model_id if request and request.model_id else "default"
    provider_id = request.provider_id if request and request.provider_id else "agentpool"

    now = now_ms()
    # Create assistant message for the summary (marked with summary=true)
    assistant_msg_id = identifier.ascending("message")
    assistant_message = AssistantMessage(
        id=assistant_msg_id,
        session_id=session_id,
        parent_id="",
        model_id=model_id,
        provider_id=provider_id,
        mode="summarize",
        agent="summarizer",
        path=MessagePath(cwd=state.working_dir, root=state.working_dir),
        time=MessageTime(created=now, completed=None),
        tokens=Tokens(cache=TokensCache(read=0, write=0), input=0, output=0, reasoning=0),
        cost=0.0,
        summary=True,  # Mark as summary message
    )

    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)
    # Broadcast message created
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
    # Mark session as busy
    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="busy")))
    # Add step-start part
    step_start = StepStartPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
    )
    assistant_msg_with_parts.parts.append(step_start)
    await state.broadcast_event(PartUpdatedEvent.create(step_start))

    # Step 1: Stream LLM summary generation FIRST (while we have full history)
    # The LLM sees the complete conversation and generates a continuation prompt.
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    text_part: TextPart | None = None

    try:
        agent = state.agent
        # Stream events from the agent with the summarization prompt
        # This runs with FULL history - the summary is based on complete context
        async for event in agent.run_stream(SUMMARIZE_PROMPT):
            match event:
                # Text streaming start
                case PartStartEvent(part=PydanticTextPart(content=delta)):
                    response_text = delta
                    text_part = TextPart(
                        id=identifier.ascending("part"),
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

                # Stream complete - extract token usage
                case StreamCompleteEvent(message=msg) if msg and msg.usage:
                    input_tokens = msg.usage.input_tokens or 0
                    output_tokens = msg.usage.output_tokens or 0

    except Exception as e:  # noqa: BLE001
        response_text = f"Error generating summary: {e}"

    response_time = now_ms()

    # Create/update text part with final response
    if text_part is None:
        text_part = TextPart(
            id=identifier.ascending("part"),
            message_id=assistant_msg_id,
            session_id=session_id,
            text=response_text,
        )
        assistant_msg_with_parts.parts.append(text_part)
        await state.broadcast_event(PartUpdatedEvent.create(text_part))

    # Step 2: Run compaction pipeline AFTER summary is generated
    # The summary was generated with full context. Now we compact the history.
    # Final state will be: [compacted history] + [summary message]
    # The compacted history becomes the cached prefix for future LLM calls.
    try:
        from agentpool.messaging.compaction import compact_conversation, summarizing_context

        # Get the compaction pipeline from the agent pool configuration
        pipeline = None
        if state.agent.agent_pool is not None:
            pipeline = state.agent.agent_pool.compaction_pipeline

        if pipeline is None:
            # Fall back to a default summarizing pipeline
            pipeline = summarizing_context()

        # Apply the compaction pipeline (modifies agent.conversation in place)
        await compact_conversation(pipeline, state.agent.conversation)

        # Persist compacted messages to storage, replacing the old ones
        if state.pool.storage is not None:
            compacted_history = state.agent.conversation.get_history()
            await state.pool.storage.replace_conversation_messages(
                session_id,
                compacted_history,
            )

        # Update in-memory OpenCode messages list with compacted versions
        # Keep only the summary message we just created
        state.messages[session_id] = [assistant_msg_with_parts]

    except Exception:  # noqa: BLE001
        # Compaction failure is not fatal - we still have the summary
        pass

    # Add step-finish part
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
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(PartUpdatedEvent.create(step_finish))
    # Update message with completion time and tokens
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
    await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
    # Mark session as idle
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    return assistant_msg_with_parts


@router.post("/{session_id}/share")
async def share_session(
    session_id: str,
    state: StateDep,
    num_messages: int | None = None,
) -> Session:
    """Share session conversation history via OpenCode's sharing service.

    Uses the OpenCode share API to create a shareable link with the full
    conversation including messages and parts.

    Returns the updated session with the share URL.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = state.messages.get(session_id, [])

    if not messages:
        raise HTTPException(status_code=400, detail="No messages to share")

    # Apply message limit if specified
    if num_messages is not None and num_messages > 0:
        messages = messages[-num_messages:]
    # Convert our messages to OpenCode Message format
    opencode_messages: list[Message] = []
    for msg_with_parts in messages:
        info = msg_with_parts.info
        # Map role to OpenCode sharing roles
        role = info.role
        if role == "model":  # type: ignore[comparison-overlap]
            mapped_role: Literal["user", "assistant", "system"] = "assistant"
        elif role in ("user", "assistant", "system"):
            mapped_role = role
        else:
            mapped_role = "user"

        # Extract text parts
        parts = [
            MessagePart(type="text", text=part.text)
            for part in msg_with_parts.parts
            if isinstance(part, TextPart) and part.text
        ]
        if parts:
            opencode_messages.append(Message(role=mapped_role, parts=parts))
    if not opencode_messages:
        raise HTTPException(status_code=400, detail="No content to share")

    # Share via OpenCode API
    async with OpenCodeSharer() as sharer:
        result = await sharer.share_conversation(opencode_messages, title=session.title)
        share_url = result.url
    # Store the share URL in the session
    share_info = SessionShare(url=share_url)
    updated_session = session.model_copy(update={"share": share_info})
    state.sessions[session_id] = updated_session
    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
    return updated_session


class RevertRequest(OpenCodeBaseModel):
    """Request body for reverting a message."""

    message_id: str
    part_id: str | None = None


@router.post("/{session_id}/revert")
async def revert_session(session_id: str, request: RevertRequest, state: StateDep) -> Session:
    """Revert file changes and messages from a specific message.

    Removes messages from the revert point onwards and restores files to their
    state before the specified message's changes.
    """
    from agentpool_server.opencode_server.models import MessageRemovedEvent, PartRemovedEvent

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get messages for this session
    messages = state.messages.get(session_id, [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to revert")

    # Find the revert message index
    revert_index = None
    for i, msg in enumerate(messages):
        if msg.info.id == request.message_id:
            revert_index = i
            break

    if revert_index is None:
        raise HTTPException(status_code=404, detail=f"Message {request.message_id} not found")

    # Split messages: keep messages before revert point, remove from revert point onwards
    messages_to_keep = messages[:revert_index]
    messages_to_remove = messages[revert_index:]

    if not messages_to_remove:
        raise HTTPException(status_code=400, detail="No messages to revert")

    # Store removed messages for unrevert
    state.reverted_messages[session_id] = messages_to_remove

    # Update message list - keep only messages before revert point
    state.messages[session_id] = messages_to_keep

    # Emit message.removed and part.removed events for all removed messages
    for msg in messages_to_remove:
        # Emit message.removed event
        await state.broadcast_event(MessageRemovedEvent.create(session_id, msg.info.id))

        # Emit part.removed events for all parts
        for part in msg.parts:
            await state.broadcast_event(PartRemovedEvent.create(session_id, msg.info.id, part.id))

    # Also revert file changes if any
    file_ops = state.pool.file_ops
    if file_ops.changes:
        revert_ops = file_ops.get_revert_operations(since_message_id=request.message_id)
        if revert_ops:
            fs = state.agent.env.get_fs()
            for path, content in revert_ops:
                try:
                    if content is None:
                        await fs._rm_file(path)
                    else:
                        content_bytes = content.encode("utf-8")
                        await fs._pipe_file(path, content_bytes)
                except Exception as e:
                    detail = f"Failed to revert {path}: {e}"
                    raise HTTPException(status_code=500, detail=detail) from e
            file_ops.remove_changes_since_message(request.message_id)

    # Update session with revert info
    session = state.sessions[session_id]
    revert_info = SessionRevert(message_id=request.message_id, part_id=request.part_id)
    updated_session = session.model_copy(update={"revert": revert_info})
    state.sessions[session_id] = updated_session

    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
    return updated_session


@router.post("/{session_id}/unrevert")
async def unrevert_session(session_id: str, state: StateDep) -> Session:
    """Restore all reverted messages and file changes.

    Re-applies the messages and changes that were previously reverted.
    """
    from agentpool_server.opencode_server.models import MessageUpdatedEvent, PartUpdatedEvent

    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Restore reverted messages
    reverted_messages = state.reverted_messages.get(session_id, [])
    if not reverted_messages:
        raise HTTPException(status_code=400, detail="No reverted messages to restore")

    # Restore messages to conversation
    if session_id not in state.messages:
        state.messages[session_id] = []
    state.messages[session_id].extend(reverted_messages)

    # Emit message.updated and part.updated events for restored messages
    for msg in reverted_messages:
        # Emit message.updated event
        await state.broadcast_event(MessageUpdatedEvent.create(msg.info))

        # Emit part.updated events for all parts
        for part in msg.parts:
            await state.broadcast_event(PartUpdatedEvent.create(part))

    # Clear reverted messages
    state.reverted_messages.pop(session_id, None)

    # Also unrevert file changes if any
    file_ops = state.pool.file_ops
    if file_ops.reverted_changes:
        unrevert_ops = file_ops.get_unrevert_operations()
        fs = state.agent.env.get_fs()
        for path, content in unrevert_ops:
            try:
                if content is None:
                    await fs._rm_file(path)
                else:
                    content_bytes = content.encode("utf-8")
                    await fs._pipe_file(path, content_bytes)
            except Exception as e:
                detail = f"Failed to unrevert {path}: {e}"
                raise HTTPException(status_code=500, detail=detail) from e
        file_ops.restore_reverted_changes()

    # Clear revert info from session
    updated_session = session.model_copy(update={"revert": None})
    state.sessions[session_id] = updated_session
    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
    return updated_session


@router.delete("/{session_id}/share")
async def unshare_session(session_id: str, state: StateDep) -> bool:
    """Remove share link from a session.

    Note: This only removes the link from the session metadata.
    The shared content may still exist on the provider's servers.
    """
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.share is None:
        raise HTTPException(status_code=400, detail="Session is not shared")
    # Remove share info from session
    updated_session = session.model_copy(update={"share": None})
    state.sessions[session_id] = updated_session
    # Broadcast session update
    await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
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
    session = await get_or_load_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    prompts = await state.agent.tools.list_prompts()
    # Find matching prompt by name
    prompt = next((p for p in prompts if p.name == request.command), None)
    if prompt is None:
        detail = f"Command not found: {request.command}"
        raise HTTPException(status_code=404, detail=detail)

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
        cost=0.0,
    )
    assistant_msg_with_parts = MessageWithParts(info=assistant_message, parts=[])
    state.messages[session_id].append(assistant_msg_with_parts)
    await state.broadcast_event(MessageUpdatedEvent.create(assistant_message))
    # Mark session as busy
    state.session_status[session_id] = SessionStatus(type="busy")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="busy")))
    # Add step-start part
    part_id = identifier.ascending("part")
    step_start = StepStartPart(id=part_id, message_id=assistant_msg_id, session_id=session_id)
    assistant_msg_with_parts.parts.append(step_start)
    await state.broadcast_event(PartUpdatedEvent.create(step_start))

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
                        if isinstance(item, FileUrl):
                            prompt_texts.append(item.url)
                        elif isinstance(item, str):
                            prompt_texts.append(item)
        prompt_text = "\n".join(prompt_texts)
        # Run the expanded prompt through the agent
        result = await state.agent.run(prompt_text)
        output_text = str(result.data)

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
    await state.broadcast_event(PartUpdatedEvent.create(text_part))
    step_finish = StepFinishPart(
        id=identifier.ascending("part"),
        message_id=assistant_msg_id,
        session_id=session_id,
        tokens=StepFinishTokens(cache=TokenCache()),
        cost=0.0,
    )
    assistant_msg_with_parts.parts.append(step_finish)
    await state.broadcast_event(PartUpdatedEvent.create(step_finish))
    # Update message with completion time
    time_ = MessageTime(created=now, completed=response_time)
    updated_assistant = assistant_message.model_copy(update={"time": time_})
    assistant_msg_with_parts.info = updated_assistant
    await state.broadcast_event(MessageUpdatedEvent.create(updated_assistant))
    # Mark session as idle
    state.session_status[session_id] = SessionStatus(type="idle")
    await state.broadcast_event(SessionStatusEvent.create(session_id, SessionStatus(type="idle")))
    return assistant_msg_with_parts
