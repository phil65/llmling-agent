"""Converters between pydantic-ai/AgentPool and OpenCode message formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import uuid

import anyenv
from pydantic import TypeAdapter
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    RetryPromptPart,
    TextPart as PydanticTextPart,
    ToolCallPart as PydanticToolCallPart,
    ToolReturnPart as PydanticToolReturnPart,
    UserPromptPart,
    VideoUrl,
)

from agentpool import log
from agentpool.common_types import PathReference
from agentpool.messaging.messages import ChatMessage
from agentpool.sessions.models import SessionData
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict
from agentpool.utils.time_utils import datetime_to_ms, ms_to_datetime
from agentpool_server.opencode_server.models import (
    APIErrorInfo,
    AssistantMessage,
    MessagePath,
    MessageTime,
    MessageWithParts,
    Model,
    ModelCost,
    ModelLimit,
    RetryPart,
    Session,
    SessionRevert,
    SessionShare,
    StepFinishPart,
    StepFinishTokens,
    StepStartPart,
    TextPart,
    TimeCreated,
    TimeCreatedUpdated,
    TimeStart,
    TimeStartEnd,
    TimeStartEndCompacted,
    TimeStartEndOptional,
    TokenCache,
    Tokens,
    TokensCache,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec.asyn import AsyncFileSystem
    from pydantic_ai import (
        MultiModalContent,
        UserContent,
    )
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from agentpool_server.opencode_server.models import Part, ToolState


logger = log.get_logger(__name__)

# Parameter name mapping from snake_case to camelCase for OpenCode TUI compatibility
_PARAM_NAME_MAP: dict[str, str] = {
    "path": "filePath",
    "file_path": "filePath",
    "old_string": "oldString",
    "new_string": "newString",
    "replace_all": "replaceAll",
    "line_hint": "lineHint",
}


def _convert_params_for_ui(params: dict[str, Any]) -> dict[str, Any]:
    """Convert parameter names from snake_case to camelCase for OpenCode TUI.

    OpenCode TUI expects camelCase parameter names like 'filePath', 'oldString', etc.
    This converts our snake_case parameters to match those expectations.
    """
    return {_PARAM_NAME_MAP.get(k, k): v for k, v in params.items()}


def generate_part_id() -> str:
    """Generate a unique part ID."""
    return str(uuid.uuid4())


def _get_input_from_state(state: ToolState, *, convert_params: bool = False) -> dict[str, Any]:
    """Extract input from any tool state type.

    Args:
        state: Tool state to extract input from
        convert_params: If True, convert param names to camelCase for UI display
    """
    if hasattr(state, "input") and state.input is not None:
        return _convert_params_for_ui(state.input) if convert_params else state.input
    return {}


def _convert_file_part_to_user_content(
    part: dict[str, Any],
    fs: AsyncFileSystem | None = None,
) -> UserContent | PathReference:
    """Convert an OpenCode FilePartInput to pydantic-ai content or PathReference.

    Supports:
    - file:// URLs with text/* or directory MIME -> PathReference (deferred resolution)
    - data: URIs -> BinaryContent
    - Images (image/*) -> ImageUrl or BinaryContent
    - Documents (application/pdf) -> DocumentUrl or BinaryContent
    - Audio (audio/*) -> AudioUrl or BinaryContent
    - Video (video/*) -> VideoUrl or BinaryContent

    Args:
        part: OpenCode file part with mime, url, and optional filename
        fs: Optional async filesystem for PathReference resolution

    Returns:
        Appropriate pydantic-ai content type or PathReference
    """
    from urllib.parse import unquote, urlparse

    mime = part.get("mime", "")
    url = part.get("url", "")
    filename = part.get("filename", "")
    # Handle data: URIs - convert to BinaryContent
    if url.startswith("data:"):
        return BinaryContent.from_data_uri(url)

    # Handle file:// URLs for text files and directories -> PathReference
    if url.startswith("file://"):
        parsed = urlparse(url)
        path = unquote(parsed.path)
        # Text files and directories get deferred context resolution
        if mime.startswith("text/") or mime == "application/x-directory" or not mime:
            return PathReference(
                path=path,
                fs=fs,
                mime_type=mime or None,
                display_name=f"@{filename}" if filename else None,
            )

        # Media files from local filesystem - use URL types
        if content := get_file_url_obj(url, mime):
            return content
        # Unknown MIME for file:// - defer to PathReference
        return PathReference(
            path=path,
            fs=fs,
            mime_type=mime or None,
            display_name=f"@{filename}" if filename else None,
        )
    # Handle regular URLs based on mime type. Fallback: treat as document
    return content if (content := get_file_url_obj(url, mime)) else DocumentUrl(url=url)


def get_file_url_obj(url: str, mime: str) -> MultiModalContent | None:
    if mime.startswith("image/"):
        return ImageUrl(url=url, media_type=mime)
    if mime.startswith("audio/"):
        return AudioUrl(url=url, media_type=mime)
    if mime.startswith("video/"):
        return VideoUrl(url=url, media_type=mime)
    if mime == "application/pdf":
        return DocumentUrl(url=url, media_type=mime)
    return None


def extract_user_prompt_from_parts(
    parts: list[dict[str, Any]],
    fs: AsyncFileSystem | None = None,
) -> str | Sequence[UserContent | PathReference]:
    """Extract user prompt from OpenCode message parts.

    Converts OpenCode parts to pydantic-ai UserContent or PathReference format:
    - Text parts become strings
    - File parts with file:// URLs become PathReference (deferred resolution)
    - Other file parts become ImageUrl, DocumentUrl, AudioUrl, VideoUrl, or BinaryContent

    Args:
        parts: List of OpenCode message parts
        fs: Optional async filesystem for PathReference resolution

    Returns:
        Either a simple string (text-only) or a list of UserContent/PathReference items
    """
    result: list[UserContent | PathReference] = []
    for part in parts:
        match part.get("type"):
            case "text" if text := part.get("text", ""):
                result.append(text)
            case "file":
                content = _convert_file_part_to_user_content(part, fs=fs)
                result.append(content)
            case "agent" if agent_name := part.get("name", ""):
                # Agent mention - inject instruction to delegate to sub-agent
                # This mirrors OpenCode's server-side behavior: inject a synthetic
                # text instruction telling the LLM to call the task tool
                # TODO: Implement proper agent delegation via task tool
                # For now, we add the instruction as text that the LLM will see
                instruction = (
                    f"Use the above message and context to generate a prompt "
                    f"and call the task tool with subagent: {agent_name}"
                )
                result.append(instruction)
            case "snapshot":
                # File system snapshot reference
                # TODO: Implement snapshot restoration/reference
                snapshot_id = part.get("snapshot", "")
                logger.debug("Ignoring snapshot part", snapshot_id=snapshot_id)
            case "patch":
                # Diff/patch content
                # TODO: Implement patch application
                patch_hash = part.get("hash", "")
                files = part.get("files", [])
                logger.debug("Ignoring patch part", patch_hash=patch_hash, files=files)
            case "reasoning" if reasoning_text := part.get("text", ""):
                # Extended thinking/reasoning content from the model
                # Include as text context since it contains useful reasoning
                result.append(f"[Reasoning]: {reasoning_text}")
            case "compaction":
                # Marks where conversation was compacted
                # TODO: Handle compaction markers for context management
                auto = part.get("auto", False)
                logger.debug("Ignoring compaction part", auto=auto)
            case "subtask":
                # References a spawned subtask
                # TODO: Implement subtask tracking/results
                agent = part.get("agent", "")
                desc = part.get("description", "")
                logger.debug("Ignoring subtask part", agent=agent, description=desc)
            case "retry":
                # Marks a retry of a failed operation
                # TODO: Handle retry tracking
                attempt = part.get("attempt", 0)
                logger.debug("Ignoring retry part", attempt=attempt)
            case "step-start":
                # Step start marker - informational only
                logger.debug("Ignoring step-start part")
            case "step-finish":
                # Step finish marker - informational only
                logger.debug("Ignoring step-finish part")
            case _ as part_type:
                # Unknown part type
                logger.warning("Unknown part type", part_type=part_type)

    # If only text parts, join them as a single string for simplicity
    if all(isinstance(item, str) for item in result):
        return "\n".join(result)  # type: ignore[arg-type]

    return result


# =============================================================================
# ChatMessage <-> OpenCode MessageWithParts Converters
# =============================================================================


def chat_message_to_opencode(  # noqa: PLR0915
    msg: ChatMessage[Any],
    session_id: str,
    working_dir: str = "",
    agent_name: str = "default",
    model_id: str = "unknown",
    provider_id: str = "agentpool",
) -> MessageWithParts:
    """Convert a ChatMessage to OpenCode MessageWithParts.

    Args:
        msg: The ChatMessage to convert
        session_id: OpenCode session ID
        working_dir: Working directory for path context
        agent_name: Name of the agent
        model_id: Model identifier
        provider_id: Provider identifier

    Returns:
        OpenCode MessageWithParts with appropriate info and parts
    """
    message_id = msg.message_id
    created_ms = datetime_to_ms(msg.timestamp)
    parts: list[Part] = []
    # Track tool calls by ID for pairing with returns
    tool_calls: dict[str, ToolPart] = {}
    if msg.role == "user":
        # User message - don't set model (users don't use models)
        info: UserMessage | AssistantMessage = UserMessage(
            id=message_id,
            session_id=session_id,
            time=TimeCreated(created=created_ms),
            agent=agent_name,
            # model=UserMessageModel(provider_id=provider_id, model_id=model_id),
        )

        # Extract text from user message
        # First try msg.content directly (simple case)
        if msg.content and isinstance(msg.content, str):
            parts.append(
                TextPart(
                    id=generate_part_id(),
                    message_id=message_id,
                    session_id=session_id,
                    text=msg.content,
                    time=TimeStartEndOptional(start=created_ms),
                )
            )
        else:
            # Fall back to extracting from messages (pydantic-ai format)
            for model_msg in msg.messages:
                if isinstance(model_msg, ModelRequest):
                    for part in model_msg.parts:
                        if isinstance(part, UserPromptPart):
                            content = part.content
                            if isinstance(content, str):
                                text = content
                            else:
                                # Multi-modal content - extract text parts
                                text = " ".join(str(c) for c in content if isinstance(c, str))
                            if text:
                                parts.append(
                                    TextPart(
                                        id=generate_part_id(),
                                        message_id=message_id,
                                        session_id=session_id,
                                        text=text,
                                        time=TimeStartEndOptional(start=created_ms),
                                    )
                                )
                elif isinstance(model_msg, dict) and model_msg.get("kind") == "request":
                    # Handle serialized dict format from storage
                    for part in model_msg.get("parts", []):
                        if part.get("part_kind") == "user-prompt":
                            text = part.get("content", "")
                            if text and isinstance(text, str):
                                parts.append(
                                    TextPart(
                                        id=generate_part_id(),
                                        message_id=message_id,
                                        session_id=session_id,
                                        text=text,
                                        time=TimeStartEndOptional(start=created_ms),
                                    )
                                )
    else:
        # Assistant message
        completed_ms = created_ms
        if msg.response_time:
            completed_ms = created_ms + int(msg.response_time * 1000)

        # Extract token usage (handle both object and dict formats)
        usage = msg.usage
        if usage:
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                cache_read = usage.get("cache_read_tokens", 0) or 0
                cache_write = usage.get("cache_write_tokens", 0) or 0
            else:
                input_tokens = usage.input_tokens or 0
                output_tokens = usage.output_tokens or 0
                cache_read = usage.cache_read_tokens or 0
                cache_write = usage.cache_write_tokens or 0
        else:
            input_tokens = output_tokens = cache_read = cache_write = 0
        cache = TokensCache(read=cache_read, write=cache_write)
        tokens = Tokens(input=input_tokens, output=output_tokens, reasoning=0, cache=cache)
        info = AssistantMessage(
            id=message_id,
            session_id=session_id,
            parent_id="",  # Would need to track parent user message
            model_id=msg.model_name or model_id,
            provider_id=msg.provider_name or provider_id,
            mode="default",
            agent=agent_name,
            path=MessagePath(cwd=working_dir, root=working_dir),
            time=MessageTime(created=created_ms, completed=completed_ms),
            tokens=tokens,
            cost=float(msg.cost_info.total_cost) if msg.cost_info else 0.0,
            finish=msg.finish_reason,
        )

        # Add step start
        part_id = generate_part_id()
        parts.append(StepStartPart(id=part_id, message_id=message_id, session_id=session_id))
        # Process all model messages to extract parts
        # Deserialize dicts to proper pydantic-ai objects if needed
        adapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)
        for raw_msg in msg.messages:
            # Deserialize dict to proper ModelRequest/ModelResponse if needed
            model_msg = adapter.validate_python(raw_msg) if isinstance(raw_msg, dict) else raw_msg
            if isinstance(model_msg, ModelResponse):
                for p in model_msg.parts:
                    if isinstance(p, PydanticTextPart):
                        parts.append(
                            TextPart(
                                id=p.id or generate_part_id(),
                                message_id=message_id,
                                session_id=session_id,
                                text=p.content,
                                time=TimeStartEndOptional(start=created_ms, end=completed_ms),
                            )
                        )
                    elif isinstance(p, PydanticToolCallPart):
                        # Create tool part in pending/running state
                        tool_input = _convert_params_for_ui(safe_args_as_dict(p))
                        tool_part = ToolPart(
                            id=generate_part_id(),
                            message_id=message_id,
                            session_id=session_id,
                            tool=p.tool_name,
                            call_id=p.tool_call_id or generate_part_id(),
                            state=ToolStateRunning(
                                time=TimeStart(start=created_ms),
                                input=tool_input,
                                title=f"Running {p.tool_name}",
                            ),
                        )
                        tool_calls[p.tool_call_id or ""] = tool_part
                        parts.append(tool_part)

            elif isinstance(model_msg, ModelRequest):
                # Check for tool returns and retries in requests (they come after responses)
                for part in model_msg.parts:
                    if isinstance(part, RetryPromptPart):
                        # Track retry attempts - count RetryPromptParts in message history
                        retry_count = sum(
                            1
                            for m in msg.messages
                            if isinstance(m, ModelRequest)
                            for p in m.parts
                            if isinstance(p, RetryPromptPart)
                        )

                        # Create error info from retry content
                        error_message = part.model_response()
                        # Try to extract more info if we have structured error details
                        is_retryable = True
                        if isinstance(part.content, list):
                            # Validation errors - always retryable
                            error_type = "validation_error"
                        elif part.tool_name:
                            # Tool-related retry
                            error_type = "tool_error"
                        else:
                            # Generic retry
                            error_type = "retry"

                        api_error = APIErrorInfo(
                            message=error_message,
                            status_code=None,  # Not available from pydantic-ai
                            is_retryable=is_retryable,
                            metadata={"error_type": error_type} if error_type else None,
                        )

                        parts.append(
                            RetryPart(
                                id=generate_part_id(),
                                message_id=message_id,
                                session_id=session_id,
                                attempt=retry_count,
                                error=api_error,
                                time=TimeCreated(created=int(part.timestamp.timestamp() * 1000)),
                            )
                        )

                    elif isinstance(part, PydanticToolReturnPart):
                        call_id = part.tool_call_id or ""
                        existing = tool_calls.get(call_id)

                        # Format output
                        tool_content = part.content
                        if isinstance(tool_content, str):
                            output = tool_content
                        elif isinstance(tool_content, dict):
                            output = anyenv.dump_json(tool_content, indent=True)
                        else:
                            output = str(tool_content) if tool_content is not None else ""
                        if existing:
                            # Update existing tool part with completion
                            existing_input = _get_input_from_state(existing.state)
                            if isinstance(tool_content, dict) and "error" in tool_content:
                                existing.state = ToolStateError(
                                    error=str(tool_content.get("error", "Unknown error")),
                                    input=existing_input,
                                    time=TimeStartEnd(start=created_ms, end=completed_ms),
                                )
                            else:
                                existing.state = ToolStateCompleted(
                                    title=f"Completed {part.tool_name}",
                                    input=existing_input,
                                    output=output,
                                    time=TimeStartEndCompacted(start=created_ms, end=completed_ms),
                                )
                        else:
                            # Orphan return - create completed tool part
                            state: ToolStateCompleted | ToolStateError
                            if isinstance(tool_content, dict) and "error" in tool_content:
                                state = ToolStateError(
                                    error=str(tool_content.get("error", "Unknown error")),
                                    input={},
                                    time=TimeStartEnd(start=created_ms, end=completed_ms),
                                )
                            else:
                                state = ToolStateCompleted(
                                    title=f"Completed {part.tool_name}",
                                    input={},
                                    output=output,
                                    time=TimeStartEndCompacted(start=created_ms, end=completed_ms),
                                )
                            parts.append(
                                ToolPart(
                                    id=generate_part_id(),
                                    message_id=message_id,
                                    session_id=session_id,
                                    tool=part.tool_name,
                                    call_id=call_id,
                                    state=state,
                                )
                            )

        # Add step finish
        parts.append(
            StepFinishPart(
                id=generate_part_id(),
                message_id=message_id,
                session_id=session_id,
                reason=msg.finish_reason or "stop",
                cost=float(msg.cost_info.total_cost) if msg.cost_info else 0.0,
                tokens=StepFinishTokens(
                    input=tokens.input,
                    output=tokens.output,
                    reasoning=tokens.reasoning,
                    cache=TokenCache(read=tokens.cache.read, write=tokens.cache.write),
                ),
            )
        )

    return MessageWithParts(info=info, parts=parts)


def opencode_to_chat_message(
    msg: MessageWithParts,
    session_id: str | None = None,
) -> ChatMessage[str]:
    """Convert OpenCode MessageWithParts to ChatMessage.

    Args:
        msg: OpenCode message with parts
        session_id: Optional conversation ID override

    Returns:
        ChatMessage with pydantic-ai model messages
    """
    info = msg.info
    message_id = info.id
    session_id = info.session_id
    # Determine role and extract timing
    if isinstance(info, UserMessage):
        role = "user"
        created_ms = info.time.created
        model_name = info.model.model_id if info.model else None
        provider_name = info.model.provider_id if info.model else None
        usage = RequestUsage()
        finish_reason = None
    else:
        role = "assistant"
        created_ms = info.time.created
        model_name = info.model_id
        provider_name = info.provider_id
        usage = RequestUsage(
            input_tokens=info.tokens.input,
            output_tokens=info.tokens.output,
            cache_read_tokens=info.tokens.cache.read,
            cache_write_tokens=info.tokens.cache.write,
        )
        finish_reason = info.finish

    timestamp = ms_to_datetime(created_ms)
    # Build model messages from parts
    model_messages: list[ModelRequest | ModelResponse] = []
    if role == "user":
        # Collect text parts into a user prompt
        text_content = [part.text for part in msg.parts if isinstance(part, TextPart)]
        content = "\n".join(text_content) if text_content else ""
        model_messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
    else:
        # Assistant message - collect response parts and tool interactions
        response_parts: list[Any] = []
        tool_returns: list[PydanticToolReturnPart] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                response_parts.append(PydanticTextPart(content=part.text, id=part.id))
            elif isinstance(part, ToolPart):
                # Create tool call part

                tool_input = _get_input_from_state(part.state)
                response_parts.append(
                    PydanticToolCallPart(
                        tool_name=part.tool,
                        tool_call_id=part.call_id,
                        args=tool_input,
                    )
                )

                # If completed/error, also create tool return
                if isinstance(part.state, ToolStateCompleted):
                    tool_returns.append(
                        PydanticToolReturnPart(
                            tool_name=part.tool,
                            tool_call_id=part.call_id,
                            content=part.state.output,
                        )
                    )
                elif isinstance(part.state, ToolStateError):
                    tool_returns.append(
                        PydanticToolReturnPart(
                            tool_name=part.tool,
                            tool_call_id=part.call_id,
                            content={"error": part.state.error},
                        )
                    )
            # Skip StepStartPart, StepFinishPart, FilePart for now

        if response_parts:
            model_messages.append(
                ModelResponse(
                    parts=response_parts,
                    usage=usage,
                    model_name=model_name,
                    timestamp=timestamp,
                )
            )

        # Add tool returns as a follow-up request if any
        if tool_returns:
            model_messages.append(ModelRequest(parts=tool_returns, instructions=None))
    # Extract content for the ChatMessage
    content = next((p.text for p in msg.parts if isinstance(p, TextPart)), "")
    return ChatMessage(
        content=content,
        role=role,  # type: ignore[arg-type]
        message_id=message_id,
        session_id=session_id or session_id,
        timestamp=timestamp,
        messages=model_messages,
        usage=usage,
        model_name=model_name,
        provider_name=provider_name,
        finish_reason=finish_reason,  # type: ignore[arg-type]
    )


def convert_toko_model_to_opencode(model: TokoModelInfo) -> Model:
    """Convert a tokonomics ModelInfo to an OpenCode Model."""
    # Convert pricing (tokonomics uses per-token, OpenCode uses per-million-token)
    input_cost = 0.0
    output_cost = 0.0
    cache_read = None
    cache_write = None

    if model.pricing:
        # tokonomics pricing is per-token, convert to per-million-tokens
        if model.pricing.prompt is not None:
            input_cost = model.pricing.prompt * 1_000_000
        if model.pricing.completion is not None:
            output_cost = model.pricing.completion * 1_000_000
        if model.pricing.input_cache_read is not None:
            cache_read = model.pricing.input_cache_read * 1_000_000
        if model.pricing.input_cache_write is not None:
            cache_write = model.pricing.input_cache_write * 1_000_000

    cost = ModelCost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read,
        cache_write=cache_write,
    )
    # Convert limits
    context = float(model.context_window) if model.context_window else 128000.0
    output = float(model.max_output_tokens) if model.max_output_tokens else 4096.0
    limit = ModelLimit(context=context, output=output)
    # Determine capabilities from modalities and metadata
    has_vision = "image" in model.input_modalities
    has_reasoning = "reasoning" in model.output_modalities or "thinking" in model.name.lower()
    # Format release date if available
    release_date = ""
    if model.created_at:
        release_date = model.created_at.strftime("%Y-%m-%d")
    # Use id_override if available (e.g., "opus" for Claude Code SDK)
    model_id = model.id_override or model.id
    return Model(
        id=model_id,
        name=model.name,
        attachment=has_vision,
        cost=cost,
        limit=limit,
        reasoning=has_reasoning,
        release_date=release_date,
        temperature=True,
        tool_call=True,  # Assume most models support tool calling
    )


# =============================================================================
# Session Converters
# =============================================================================


def session_data_to_opencode(data: SessionData) -> Session:
    """Convert SessionData to OpenCode Session model.

    Args:
        data: SessionData to convert (title comes from data.title property)
    """
    # Convert datetime to milliseconds timestamp
    created_ms = datetime_to_ms(data.created_at)
    updated_ms = datetime_to_ms(data.last_active)
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
        title=data.title or "New Session",
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
    # Store revert/share in metadata
    metadata: dict[str, Any] = {}
    if session.revert:
        metadata["revert"] = session.revert.model_dump()
    if session.share:
        metadata["share"] = session.share.model_dump()
    return SessionData(
        session_id=session.id,
        agent_name=agent_name,
        pool_id=pool_id,
        project_id=session.project_id,
        parent_id=session.parent_id,
        version=session.version,
        cwd=session.directory,
        created_at=ms_to_datetime(session.time.created),
        last_active=ms_to_datetime(session.time.updated),
        metadata=metadata,
    )


def agent_messages_to_opencode(
    chat_messages: list[ChatMessage[Any]],
    session_id: str,
    working_dir: str,
    agent_name: str,
) -> list[MessageWithParts]:
    """Convert agent's chat messages to OpenCode format.

    Args:
        chat_messages: List of ChatMessage from agent.conversation
        session_id: OpenCode session ID
        working_dir: Working directory for path context
        agent_name: Name of the agent

    Returns:
        List of OpenCode MessageWithParts
    """
    opencode_messages: list[MessageWithParts] = []
    for chat_msg in chat_messages:
        opencode_msg = chat_message_to_opencode(
            chat_msg,
            session_id=session_id,
            working_dir=working_dir,
            agent_name=agent_name,
            model_id=chat_msg.model_name or "sonnet",  # Normalized name from Claude storage
            provider_id=chat_msg.provider_name or "claude-code",
        )
        opencode_messages.append(opencode_msg)
    return opencode_messages
