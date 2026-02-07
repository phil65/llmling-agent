"""Converters between pydantic-ai/AgentPool and OpenCode message formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyenv
from pydantic_ai import (
    BinaryContent,
    DocumentUrl,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    RetryPromptPart,
    TextPart as PydanticTextPart,
    ToolCallPart as PydanticToolCallPart,
    ToolReturnPart as PydanticToolReturnPart,
    UserPromptPart,
)

from agentpool import log
from agentpool.common_types import PathReference
from agentpool.messaging.messages import ChatMessage
from agentpool.sessions.models import SessionData
from agentpool.utils.pydantic_ai_helpers import get_file_url_obj, safe_args_as_dict
from agentpool.utils.time_utils import datetime_to_ms, ms_to_datetime
from agentpool_server.opencode_server.models import (
    AgentPartInput,
    FilePartInput,
    MessagePath,
    MessageTime,
    MessageWithParts,
    Session,
    SessionRevert,
    SessionShare,
    SubtaskPartInput,
    TextPart,
    TextPartInput,
    TimeCreated,
    TimeCreatedUpdated,
    TimeStart,
    TimeStartEnd,
    TimeStartEndCompacted,
    TimeStartEndOptional,
    TokenCache,
    Tokens,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
    ToolStateRunning,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec.asyn import AsyncFileSystem
    from pydantic_ai import UserContent

    from agentpool.tools.manager import ToolManager
    from agentpool_server.opencode_server.models import ToolState
    from agentpool_server.opencode_server.models.message import PartInput
    from agentpool_server.opencode_server.models.parts import ResourceSource


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


def _get_input_from_state(state: ToolState, *, convert_params: bool = False) -> dict[str, Any]:
    """Extract input from any tool state type.

    Args:
        state: Tool state to extract input from
        convert_params: If True, convert param names to camelCase for UI display
    """
    return _convert_params_for_ui(state.input) if convert_params else state.input


def _convert_file_part_to_user_content(
    mime: str,
    url: str,
    filename: str | None = None,
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
        mime: Mime type
        url: part URL
        filename: Optional filename
        fs: Optional async filesystem for PathReference resolution

    Returns:
        Appropriate pydantic-ai content type or PathReference
    """
    from urllib.parse import unquote, urlparse

    # Handle data: URIs - convert to BinaryContent
    if url.startswith("data:"):
        return BinaryContent.from_data_uri(url)

    # Handle file:// URLs for text files and directories -> PathReference
    if url.startswith("file://"):
        parsed = urlparse(url)
        path = unquote(parsed.path)
        # Text files and directories get deferred context resolution
        title = f"@{filename}" if filename else None
        if mime.startswith("text/") or mime == "application/x-directory" or not mime:
            return PathReference(path=path, fs=fs, mime_type=mime or None, display_name=title)

        # Media files from local filesystem - use URL types
        if content := get_file_url_obj(url, mime):
            return content
        # Unknown MIME for file:// - defer to PathReference
        return PathReference(path=path, fs=fs, mime_type=mime or None, display_name=title)
    # Handle regular URLs based on mime type. Fallback: treat as document
    return content if (content := get_file_url_obj(url, mime)) else DocumentUrl(url=url)


async def _resolve_mcp_resource(source: ResourceSource, tools: ToolManager) -> str | None:
    """Resolve an MCP resource and return its content as text (or None if cant be read)."""
    try:
        resources = await tools.list_resources()
        resource = next(
            (r for r in resources if r.uri == source.uri and r.client == source.client_name),
            None,
        )
        if resource is None:
            logger.warning("MCP resource not found: %s/%s", source.client_name, source.uri)
            return None
        contents = await resource.read()
        return "\n".join(contents) if contents else None
    except Exception:
        logger.exception("Failed to read MCP resource: %s/%s", source.client_name, source.uri)
        return None


async def extract_user_prompt_from_parts(
    parts: list[PartInput],
    fs: AsyncFileSystem | None = None,
    tools: ToolManager | None = None,
) -> Sequence[UserContent | PathReference]:
    """Extract user prompt from OpenCode message input parts.

    Converts OpenCode input parts to pydantic-ai UserContent or PathReference format:
    - Text parts become strings
    - File parts with file:// URLs become PathReference (deferred resolution)
    - File parts with ResourceSource are resolved via MCP
    - Other file parts become ImageUrl, DocumentUrl, AudioUrl, VideoUrl, or BinaryContent
    - Agent parts inject instructions to delegate to sub-agents
    - Subtask parts inject instructions for spawning subtasks

    Args:
        parts: List of OpenCode message input parts
        fs: Optional async filesystem for PathReference resolution
        tools: Optional tool manager for resolving MCP resources

    Returns:
        Either a simple string (text-only) or a list of UserContent/PathReference items
    """
    from agentpool_server.opencode_server.models.parts import ResourceSource

    result: list[UserContent | PathReference] = []
    for part in parts:
        match part:
            case TextPartInput(text=text) if text:
                result.append(text)
            case FilePartInput(source=ResourceSource() as resource) if tools is not None:
                content = await _resolve_mcp_resource(resource, tools)
                if content is not None:
                    result.append(content)
            case FilePartInput(mime=mime, url=url, filename=filename):
                file_content = _convert_file_part_to_user_content(mime, url, filename, fs=fs)
                result.append(file_content)
            case AgentPartInput(name=agent_name):
                # Agent mention (@agent-name in prompt) - inject instruction to execute task
                # This mirrors OpenCode's server-side behavior: inject a synthetic
                # text instruction telling the LLM to call the task tool
                instruction = (
                    f"Use the above message and context to generate a prompt "
                    f"and call the task tool with agent_or_team='{agent_name}'"
                )
                result.append(instruction)
            case SubtaskPartInput(agent=agent, prompt=subtask_prompt, description=desc):
                # Subtask - explicit task execution with pre-defined prompt
                # Inject instruction to call task with the provided parameters
                instruction = (
                    f"Call the task tool with:\n"
                    f"  agent_or_team: '{agent}'\n"
                    f"  prompt: '{subtask_prompt}'\n"
                    f"  description: '{desc}'"
                )
                result.append(instruction)

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
    if msg.role == "user":
        result = MessageWithParts.user(
            message_id=message_id,
            session_id=session_id,
            time=TimeCreated(created=created_ms),
            agent_name=agent_name,
        )
        if msg.content and isinstance(msg.content, str):
            ts_opt = TimeStartEndOptional(start=created_ms)
            result.add_text_part(msg.content, time=ts_opt)
        else:
            for model_msg in msg.messages:
                if not isinstance(model_msg, ModelRequest):
                    continue
                for part in model_msg.parts:
                    if not isinstance(part, UserPromptPart):
                        continue
                    content = part.content
                    if isinstance(content, str):
                        text = content
                    else:
                        text = " ".join(str(c) for c in content if isinstance(c, str))
                    if text:
                        ts_opt = TimeStartEndOptional(start=created_ms)
                        result.add_text_part(text, time=ts_opt)
    else:
        # Assistant message
        completed_ms = created_ms
        if msg.response_time:
            completed_ms = created_ms + int(msg.response_time * 1000)

        usage = msg.usage
        cache = TokenCache(read=usage.cache_read_tokens, write=usage.cache_write_tokens)
        tokens = Tokens(input=usage.input_tokens, output=usage.output_tokens, cache=cache)
        result = MessageWithParts.assistant(
            message_id=message_id,
            session_id=session_id,
            parent_id="",  # Would need to track parent user message
            model_id=msg.model_name or model_id,
            provider_id=msg.provider_name or provider_id,
            mode="default",
            agent_name=agent_name,
            path=MessagePath(cwd=working_dir, root=working_dir),
            time=MessageTime(created=created_ms, completed=completed_ms),
            tokens=tokens,
            cost=float(msg.cost_info.total_cost) if msg.cost_info else 0.0,
            finish=msg.finish_reason,
        )

        result.add_step_start_part()
        # Process all model messages to extract parts
        tool_calls: dict[str, ToolPart] = {}
        for model_msg in msg.messages:
            for p in model_msg.parts:
                match p:
                    case PydanticTextPart(content=content):
                        ts_opt = TimeStartEndOptional(start=created_ms, end=completed_ms)
                        result.add_text_part(content, time=ts_opt)
                    case PydanticToolCallPart(tool_name=tool_name, tool_call_id=call_id):
                        tool_input = _convert_params_for_ui(safe_args_as_dict(p))
                        ts = TimeStart(start=created_ms)
                        title = f"Running {tool_name}"
                        running_state = ToolStateRunning(time=ts, input=tool_input, title=title)
                        tool_part = result.add_tool_part(tool_name, call_id, state=running_state)
                        tool_calls[call_id] = tool_part
                    case RetryPromptPart(content=retry_content, tool_name=tool_name, timestamp=ts):
                        retry_count = sum(
                            1
                            for m in msg.messages
                            if isinstance(m, ModelRequest)
                            for p in m.parts
                            if isinstance(p, RetryPromptPart)
                        )
                        error_message = p.model_response()
                        is_retryable = True
                        if isinstance(retry_content, list):
                            error_type = "validation_error"
                        elif tool_name:
                            error_type = "tool_error"
                        else:
                            error_type = "retry"

                        result.add_retry_part(
                            attempt=retry_count,
                            message=error_message,
                            created=int(ts.timestamp() * 1000),
                            is_retryable=is_retryable,
                            metadata={"error_type": error_type} if error_type else None,
                        )
                    case PydanticToolReturnPart(
                        tool_call_id=call_id,
                        content=tool_content,
                        tool_name=tool_name,
                        timestamp=tool_ts,
                    ):
                        end_ms = datetime_to_ms(tool_ts)
                        if isinstance(tool_content, str):
                            output = tool_content
                        elif isinstance(tool_content, dict):
                            output = anyenv.dump_json(tool_content, indent=True)
                        else:
                            output = str(tool_content) if tool_content is not None else ""
                        if existing := tool_calls.get(call_id):
                            existing_input = _get_input_from_state(existing.state)
                            if isinstance(tool_content, dict) and "error" in tool_content:
                                existing.state = ToolStateError(
                                    error=str(tool_content.get("error", "Unknown error")),
                                    input=existing_input,
                                    time=TimeStartEnd(start=created_ms, end=end_ms),
                                )
                            else:
                                title = f"Completed {tool_name}"
                                tsc = TimeStartEndCompacted(start=created_ms, end=end_ms)
                                existing.state = ToolStateCompleted(
                                    title=title, input=existing_input, output=output, time=tsc
                                )
                        else:
                            # Orphan return - create completed tool part
                            state: ToolStateCompleted | ToolStateError
                            if isinstance(tool_content, dict) and "error" in tool_content:
                                err = str(tool_content.get("error", "Unknown error"))
                                ts_end = TimeStartEnd(start=created_ms, end=end_ms)
                                state = ToolStateError(error=err, time=ts_end)
                            else:
                                title = f"Completed {tool_name}"
                                tsc = TimeStartEndCompacted(start=created_ms, end=end_ms)
                                state = ToolStateCompleted(title=title, output=output, time=tsc)
                            result.add_tool_part(tool_name, call_id, state=state)
        tokens = Tokens(
            input=tokens.input,
            output=tokens.output,
            reasoning=tokens.reasoning,
            cache=TokenCache(read=tokens.cache.read, write=tokens.cache.write),
        )
        cost = float(msg.cost_info.total_cost) if msg.cost_info else 0.0
        result.add_step_finish_part(reason=msg.finish_reason or "stop", cost=cost, tokens=tokens)

    return result


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
            match part:
                case TextPart(text=text, id=part_id):
                    response_parts.append(PydanticTextPart(content=text, id=part_id))
                case ToolPart(tool=tool_name, call_id=call_id, state=state):
                    response_parts.append(
                        PydanticToolCallPart(
                            tool_name=tool_name,
                            tool_call_id=call_id,
                            args=_get_input_from_state(state),
                        )
                    )
                    match state:
                        case ToolStateCompleted(output=output):
                            tool_returns.append(
                                PydanticToolReturnPart(
                                    tool_name=tool_name,
                                    tool_call_id=call_id,
                                    content=output,
                                )
                            )
                        case ToolStateError(error=error):
                            tool_returns.append(
                                PydanticToolReturnPart(
                                    tool_name=tool_name,
                                    tool_call_id=call_id,
                                    content={"error": error},
                                )
                            )

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
