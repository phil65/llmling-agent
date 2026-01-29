"""Convert between Codex and AgentPool types.

Provides converters for:
- Event conversion (Codex streaming events -> AgentPool events)
- MCP server configs (Native configs -> Codex types)
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, overload

from pydantic_ai import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentpool.messaging import ChatMessage
from agentpool.sessions import SessionData


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import UserContent
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool_config.mcp_server import (
        MCPServerConfig,
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )
    from codex_adapter import ModelData, ThreadData
    from codex_adapter.codex_types import HttpMcpServer, McpServerConfig, StdioMcpServer
    from codex_adapter.events import CodexEvent
    from codex_adapter.models import (
        ThreadItem,
        Turn,
        TurnInputItem,
        UserInputImage,
        UserInputLocalImage,
        UserInputSkill,
        UserInputText,
    )


@overload
def mcp_config_to_codex(config: StdioMCPServerConfig) -> tuple[str, StdioMcpServer]: ...


@overload
def mcp_config_to_codex(config: SSEMCPServerConfig) -> tuple[str, HttpMcpServer]: ...


@overload
def mcp_config_to_codex(
    config: StreamableHTTPMCPServerConfig,
) -> tuple[str, HttpMcpServer]: ...


@overload
def mcp_config_to_codex(config: MCPServerConfig) -> tuple[str, McpServerConfig]: ...


def mcp_config_to_codex(config: MCPServerConfig) -> tuple[str, McpServerConfig]:
    """Convert native MCPServerConfig to (name, Codex McpServerConfig) tuple.

    Args:
        config: Native MCP server configuration

    Returns:
        Tuple of (server name, Codex-compatible MCP server configuration)
    """
    from agentpool_config.mcp_server import (
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )
    from codex_adapter.codex_types import HttpMcpServer, StdioMcpServer

    # Name should not be None by the time we use it
    server_name = config.name or f"server_{id(config)}"
    match config:
        case StdioMCPServerConfig():
            return (
                server_name,
                StdioMcpServer(
                    command=config.command,
                    args=config.args or [],
                    env=config.env,
                    enabled=config.enabled,
                ),
            )

        case SSEMCPServerConfig():
            # Codex uses HTTP transport for SSE
            # SSE config just has URL, no separate auth fields
            return (server_name, HttpMcpServer(url=str(config.url), enabled=config.enabled))

        case StreamableHTTPMCPServerConfig():
            # StreamableHTTP has headers field
            headers = config.headers if config.headers else None
            return (
                server_name,
                HttpMcpServer(url=str(config.url), http_headers=headers, enabled=config.enabled),
            )

        case _:
            msg = f"Unsupported MCP server config type: {type(config)}"
            raise TypeError(msg)


def to_model_info(model_data: ModelData) -> TokoModelInfo:
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    model_id = model_data.model or model_data.id
    # Use display_name and description from API if available
    return TokoModelInfo(
        id=model_id,
        name=model_data.display_name or model_data.id,
        provider="openai",
        description=model_data.description or f"Model: {model_id}",
        id_override=model_id,
    )


def to_session_data(thread_data: ThreadData, agent_name: str, cwd: str | None) -> SessionData:
    created_at = datetime.fromtimestamp(thread_data.created_at, tz=UTC)
    return SessionData(
        session_id=thread_data.id,
        agent_name=agent_name,
        cwd=thread_data.cwd or cwd,
        created_at=created_at,
        last_active=created_at,  # Codex doesn't track separate last_active
        metadata={"title": thread_data.preview} if thread_data.preview else {},
    )


def user_content_to_codex(
    content: list[UserContent],
) -> list[TurnInputItem]:
    """Convert pydantic-ai UserContent list to Codex TurnInputItem list."""
    from codex_adapter.models import ImageInputItem, TextInputItem

    result: list[TurnInputItem] = []
    for item in content:
        match item:
            case str():
                result.append(TextInputItem(text=item))
            case ImageUrl():
                result.append(ImageInputItem(url=item.url))
            case BinaryContent() if item.media_type.startswith("image/"):
                # Convert binary image to data URI
                b64 = base64.b64encode(item.data).decode()
                data_uri = f"data:{item.media_type};base64,{b64}"
                result.append(ImageInputItem(url=data_uri))
            case _:
                # AudioUrl, DocumentUrl, VideoUrl, CachePoint - not supported by Codex
                pass
    return result


def _format_tool_result(item: ThreadItem) -> str:  # noqa: PLR0911
    """Format tool result from a completed ThreadItem.

    Args:
        item: Completed thread item

    Returns:
        Formatted result string
    """
    from codex_adapter.models import (
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemMcpToolCall,
    )

    match item:
        case ThreadItemCommandExecution():
            output = item.aggregated_output or ""
            if output:
                return f"```\n{output}\n```"
            return ""
        case ThreadItemFileChange():
            # Format file changes with their diffs
            parts = []
            for change in item.changes:
                kind = change.kind.kind  # "add", "delete", or "update"
                path = change.path
                parts.append(f"{kind.upper()}: {path}")
                if change.diff:
                    parts.append(change.diff)
            return "\n".join(parts)
        case ThreadItemMcpToolCall():
            if item.result and item.result.content:
                texts = [str(block.model_dump().get("text", "")) for block in item.result.content]
                return "\n".join(texts)
            if item.error:
                return f"Error: {item.error.message}"
            return ""
        case _:
            return ""


def _thread_item_to_tool_return_part(  # noqa: PLR0911
    item: ThreadItem,
) -> ToolReturnPart | BuiltinToolReturnPart | None:
    """Convert a completed ThreadItem to a ToolReturnPart or BuiltinToolReturnPart.

    Codex built-in tools (bash, file changes, web search, etc.) are converted to
    BuiltinToolReturnPart since they're provided by the remote Codex agent.
    MCP tools are converted to ToolReturnPart (they may be from local ToolBridge).

    Args:
        item: Completed thread item from Codex

    Returns:
        ToolReturnPart for MCP tools, BuiltinToolReturnPart for Codex built-ins, or None
    """
    from codex_adapter.models import (
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemImageView,
        ThreadItemMcpToolCall,
        ThreadItemWebSearch,
    )

    # Only process completed items
    if hasattr(item, "status") and item.status != "completed":  # pyright: ignore[reportAttributeAccessIssue]
        return None

    result = _format_tool_result(item)
    match item:
        case ThreadItemCommandExecution():
            return BuiltinToolReturnPart(tool_name="bash", content=result, tool_call_id=item.id)
        case ThreadItemFileChange():
            return BuiltinToolReturnPart("file_change", content=result, tool_call_id=item.id)
        case ThreadItemWebSearch():
            return BuiltinToolReturnPart("web_search", content=result, tool_call_id=item.id)
        case ThreadItemImageView():
            return BuiltinToolReturnPart("image_view", content=result, tool_call_id=item.id)
        case ThreadItemMcpToolCall():
            # TODO: Distinguish between local (ToolBridge) and remote MCP tools
            # See matching TODO in _thread_item_to_tool_call_part
            return ToolReturnPart(tool_name=item.tool, content=result, tool_call_id=item.id)
        case _:
            return None


def _thread_item_to_tool_call_part(
    item: ThreadItem,
) -> ToolCallPart | BuiltinToolCallPart | None:
    """Convert a ThreadItem to a ToolCallPart or BuiltinToolCallPart.

    Codex built-in tools (bash, file changes, web search, etc.) are converted to
    BuiltinToolCallPart since they're provided by the remote Codex agent.
    MCP tools are converted to ToolCallPart (they may be from local ToolBridge).

    Args:
        item: Thread item from Codex

    Returns:
        ToolCallPart for MCP tools, BuiltinToolCallPart for Codex built-ins, or None
    """
    from codex_adapter.models import (
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemImageView,
        ThreadItemMcpToolCall,
        ThreadItemWebSearch,
    )

    match item:
        case ThreadItemCommandExecution():
            args: dict[str, Any] = {"command": item.command, "cwd": item.cwd}
            return BuiltinToolCallPart(tool_name="bash", args=args, tool_call_id=item.id)
        case ThreadItemFileChange():
            args = {"changes": [c.model_dump() for c in item.changes]}
            return BuiltinToolCallPart(tool_name="file_change", args=args, tool_call_id=item.id)
        case ThreadItemWebSearch():
            args = {"query": item.query}
            return BuiltinToolCallPart(tool_name="web_search", args=args, tool_call_id=item.id)
        case ThreadItemImageView():
            args = {"path": item.path}
            return BuiltinToolCallPart(tool_name="image_view", args=args, tool_call_id=item.id)
        case ThreadItemMcpToolCall():
            # TODO: Distinguish between local (ToolBridge) and remote MCP tools
            # Currently all MCP tools use ToolCallPart, but ideally:
            # - Tools from AgentPool's ToolBridge → ToolCallPart (our tools)
            # - Tools from Codex's own MCP servers → BuiltinToolCallPart (their tools)
            # This requires tracking which tools came from ToolBridge vs Codex config
            args = item.arguments if isinstance(item.arguments, dict) else {"args": item.arguments}
            return ToolCallPart(tool_name=item.tool, args=args, tool_call_id=item.id)
        case _:
            return None


async def convert_codex_stream(  # noqa: PLR0915
    events: AsyncIterator[CodexEvent],
) -> AsyncIterator[RichAgentStreamEvent[Any]]:
    """Convert Codex event stream to native events with stateful accumulation.

    This async generator handles stateful conversion of Codex events, including:
    - Accumulating command execution output deltas into ToolCallProgressEvents
    - Accumulating file change output deltas into ToolCallProgressEvents
    - Simple 1:1 conversion for other event types

    Args:
        events: Async iterator of Codex events from the app-server

    Yields:
        Native AgentPool stream events
    """
    from agentpool.agents.events import (
        CompactionEvent,
        PartDeltaEvent,
        PlanUpdateEvent,
        TextContentItem,
        ToolCallCompleteEvent,
        ToolCallProgressEvent,
        ToolCallStartEvent,
    )
    from agentpool.resource_providers.plan_provider import PlanEntry
    from codex_adapter.models import (
        AgentMessageDeltaData,
        CommandExecutionOutputDeltaData,
        FileChangeOutputDeltaData,
        ItemCompletedData,
        ItemStartedData,
        McpToolCallProgressData,
        ReasoningTextDeltaData,
        ThreadCompactedData,
        ThreadItemCommandExecution,
        ThreadItemFileChange,
        ThreadItemMcpToolCall,
        TurnPlanUpdatedData,
    )

    # Accumulation state for streaming tool outputs
    tool_outputs: dict[str, list[str]] = {}

    async for event in events:
        match event.event_type:
            # === Stateful: Accumulate command execution output ===
            case "item/commandExecution/outputDelta" if isinstance(
                event.data, CommandExecutionOutputDeltaData
            ):
                item_id = event.data.item_id
                if item_id not in tool_outputs:
                    tool_outputs[item_id] = []
                tool_outputs[item_id].append(event.data.delta)

                # Emit accumulated progress with replace semantics, wrapped in code block
                output = "".join(tool_outputs[item_id])
                items = [TextContentItem(text=f"```\n{output}\n```")]
                yield ToolCallProgressEvent(tool_call_id=item_id, items=items, replace_content=True)

            # === File change output delta - ignore the summary, we show diff from item/started ===
            case "item/fileChange/outputDelta" if isinstance(event.data, FileChangeOutputDeltaData):
                # The outputDelta is just "Success. Updated..." summary - not useful
                # We already emitted the actual diff content in item/started
                pass

            # === Stateless: Text deltas from agent messages ===
            case "item/agentMessage/delta" if isinstance(event.data, AgentMessageDeltaData):
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=event.data.delta))

            # === Stateless: Reasoning/thinking deltas ===
            case "item/reasoning/textDelta" if isinstance(event.data, ReasoningTextDeltaData):
                yield PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=event.data.delta)
                )

            # === Stateless: Tool/command started ===
            case "item/started" if isinstance(event.data, ItemStartedData):
                item = event.data.item
                if part := _thread_item_to_tool_call_part(item):
                    # Extract title based on tool type
                    match item:
                        case ThreadItemCommandExecution():
                            title = f"Execute: {item.command}"
                        case ThreadItemFileChange():
                            # Build title from file paths
                            paths = [c.path for c in item.changes[:3]]  # First 3 paths
                            if len(item.changes) > 3:  # noqa: PLR2004
                                title = f"Edit: {', '.join(paths)} (+{len(item.changes) - 3} more)"
                            else:
                                title = f"Edit: {', '.join(paths)}"
                        case ThreadItemMcpToolCall():
                            title = f"Call {item.tool}"
                        case _:
                            title = f"Call {part.tool_name}"

                    yield ToolCallStartEvent(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        title=title,
                        raw_input=part.args_as_dict(),
                    )

                    # For file changes, immediately emit the diff as progress
                    if isinstance(item, ThreadItemFileChange):
                        diff_parts = []
                        for change in item.changes:
                            kind = change.kind.kind
                            path = change.path
                            diff_parts.append(f"{kind.upper()}: {path}")
                            if change.diff:
                                diff_parts.append(change.diff)
                        if diff_parts:
                            yield ToolCallProgressEvent(
                                tool_call_id=part.tool_call_id,
                                items=[TextContentItem(text="\n".join(diff_parts))],
                            )

            # === Stateful: Tool/command completed - clean up accumulator ===
            case "item/completed" if isinstance(event.data, ItemCompletedData):
                item = event.data.item
                # Clean up accumulated output for this item
                if item.id in tool_outputs:
                    del tool_outputs[item.id]

                if part := _thread_item_to_tool_call_part(item):
                    yield ToolCallCompleteEvent(
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        tool_input=part.args_as_dict(),
                        tool_result=_format_tool_result(item),
                        agent_name="codex",  # Will be overridden by agent
                        message_id=event.data.turn_id,
                    )

            # === Stateless: MCP tool call progress ===
            case "item/mcpToolCall/progress" if isinstance(event.data, McpToolCallProgressData):
                yield ToolCallProgressEvent(
                    tool_call_id=event.data.item_id,
                    message=event.data.message,
                )

            # === Stateless: Thread compacted ===
            case "thread/compacted" if isinstance(event.data, ThreadCompactedData):
                yield CompactionEvent(session_id=event.data.thread_id, phase="completed")

            # === Stateless: Turn plan updated ===
            case "turn/plan/updated" if isinstance(event.data, TurnPlanUpdatedData):
                entries = [
                    PlanEntry(
                        content=step.step,
                        priority="medium",  # Codex doesn't provide priority
                        status="in_progress" if step.status == "inProgress" else step.status,
                    )
                    for step in event.data.plan
                ]
                yield PlanUpdateEvent(entries=entries)

            # Ignore other events (token usage, turn started/completed, etc.)
            case _:
                pass


def event_to_part(
    event: CodexEvent,
) -> (
    TextPart
    | ThinkingPart
    | ToolCallPart
    | BuiltinToolCallPart
    | ToolReturnPart
    | BuiltinToolReturnPart
    | None
):
    """Convert Codex event to part for message construction.

    This is for building final messages, not for streaming.

    Handles both tool calls (item/started) and tool returns (item/completed).

    Args:
        event: Codex event

    Returns:
        Part or None
    """
    from codex_adapter.models import (
        AgentMessageDeltaData,
        ItemCompletedData,
        ItemStartedData,
        ReasoningTextDeltaData,
    )

    match event.event_type:
        case "item/agentMessage/delta" if isinstance(event.data, AgentMessageDeltaData):
            return TextPart(content=event.data.delta)
        case "item/reasoning/textDelta" if isinstance(event.data, ReasoningTextDeltaData):
            return ThinkingPart(content=event.data.delta)
        case "item/started" if isinstance(event.data, ItemStartedData):
            return _thread_item_to_tool_call_part(event.data.item)
        case "item/completed" if isinstance(event.data, ItemCompletedData):
            return _thread_item_to_tool_return_part(event.data.item)
    return None


def _user_input_to_content(
    inp: UserInputText | UserInputImage | UserInputLocalImage | UserInputSkill,
) -> UserContent:
    """Convert Codex UserInput to pydantic-ai UserContent."""
    from codex_adapter.models import (
        UserInputImage,
        UserInputLocalImage,
        UserInputSkill,
        UserInputText,
    )

    match inp:
        case UserInputText():
            return inp.text
        case UserInputImage():
            return ImageUrl(url=inp.url)
        case UserInputLocalImage():
            return ImageUrl(url=f"file://{inp.path}")
        case UserInputSkill():
            return f"[Skill: {inp.name}]"


def _turn_to_chat_messages(turn: Turn) -> list[ChatMessage[list[UserContent]]]:  # noqa: PLR0915
    """Convert one Turn to ChatMessages (user and optionally assistant).

    Each ThreadItem in the turn becomes one "conversational beat" in the assistant
    message's messages list (one ModelResponse per item).

    Args:
        turn: Single Turn from Codex thread

    Returns:
        List of ChatMessages - always includes user message, assistant message
        only if there are assistant responses (handles interrupted/incomplete turns)
    """
    from codex_adapter.models import (
        ThreadItemAgentMessage,
        ThreadItemCollabAgentToolCall,
        ThreadItemCommandExecution,
        ThreadItemEnteredReviewMode,
        ThreadItemExitedReviewMode,
        ThreadItemFileChange,
        ThreadItemImageView,
        ThreadItemMcpToolCall,
        ThreadItemReasoning,
        ThreadItemUserMessage,
        ThreadItemWebSearch,
    )

    user_content: list[UserContent] = []
    assistant_responses: list[ModelRequest | ModelResponse] = []  # One per ThreadItem
    assistant_display_parts: list[str] = []

    for item in turn.items:
        match item:
            case ThreadItemUserMessage():
                user_content.extend(_user_input_to_content(i) for i in item.content)
            case ThreadItemAgentMessage():
                assistant_responses.append(ModelResponse(parts=[TextPart(content=item.text)]))
                assistant_display_parts.append(item.text)
            case ThreadItemReasoning():
                # summary is list[str] - create one ThinkingPart per summary item
                # But we want one ModelResponse per ThreadItem, so combine them
                thinking_parts = [ThinkingPart(content=s) for s in item.summary]
                assistant_responses.append(ModelResponse(parts=thinking_parts))
            case ThreadItemCommandExecution():
                cmd = item.command
                output = item.aggregated_output or ""
                display = f"[Executed: {cmd}]" + (f"\n{output[:200]}" if output else "")
                assistant_display_parts.append(display)
                cmd_args: dict[str, str] = {"command": cmd}
                if item.cwd:
                    cmd_args["cwd"] = item.cwd
                bash_call = BuiltinToolCallPart(
                    tool_name="bash", args=cmd_args, tool_call_id=item.id
                )
                bash_ret = ToolReturnPart(tool_name="bash", content=output, tool_call_id=item.id)
                assistant_responses.append(ModelResponse(parts=[bash_call]))
                assistant_responses.append(ModelRequest(parts=[bash_ret]))

            case ThreadItemFileChange():
                paths = [c.path for c in item.changes]
                if len(paths) > 3:  # noqa: PLR2004
                    display = f"[Files: {', '.join(paths[:3])} +{len(paths) - 3} more]"
                else:
                    display = f"[Files: {', '.join(paths)}]"
                assistant_display_parts.append(display)
                diffs = [c.diff for c in item.changes if c.diff]
                text = "\n".join(diffs) or "OK"
                edit_call = ToolCallPart(
                    tool_name="edit", args={"files": paths}, tool_call_id=item.id
                )
                edit_ret = ToolReturnPart(tool_name="edit", content=text, tool_call_id=item.id)
                assistant_responses.append(ModelResponse(parts=[edit_call]))
                assistant_responses.append(ModelRequest(parts=[edit_ret]))

            case ThreadItemMcpToolCall():
                result_text = ""
                if item.result and item.result.content:
                    texts = [str(b.model_dump().get("text", "")) for b in item.result.content]
                    result_text = " ".join(texts)
                assistant_display_parts.append(f"[Tool: {item.tool}] {result_text[:100]}")
                mcp_args = item.arguments if isinstance(item.arguments, dict) else {}
                mcp_call = BuiltinToolCallPart(
                    tool_name=item.tool, args=mcp_args, tool_call_id=item.id
                )
                mcp_ret = ToolReturnPart(
                    tool_name=item.tool, content=result_text, tool_call_id=item.id
                )
                assistant_responses.append(ModelResponse(parts=[mcp_call]))
                assistant_responses.append(ModelRequest(parts=[mcp_ret]))

            case ThreadItemWebSearch():
                assistant_display_parts.append(f"[Web Search: {item.query}]")
                search_call = BuiltinToolCallPart(
                    tool_name="web_search", args={"query": item.query}, tool_call_id=item.id
                )
                search_ret = ToolReturnPart(
                    tool_name="web_search", content="Search completed", tool_call_id=item.id
                )
                assistant_responses.append(ModelResponse(parts=[search_call]))
                assistant_responses.append(ModelRequest(parts=[search_ret]))

            case ThreadItemImageView():
                assistant_display_parts.append(f"[Viewed Image: {item.path}]")
                view_call = BuiltinToolCallPart(
                    tool_name="view_image", args={"path": item.path}, tool_call_id=item.id
                )
                view_ret = ToolReturnPart(
                    tool_name="view_image", content="Image viewed", tool_call_id=item.id
                )
                assistant_responses.append(ModelResponse(parts=[view_call]))
                assistant_responses.append(ModelRequest(parts=[view_ret]))

            case ThreadItemEnteredReviewMode():
                assistant_display_parts.append(f"[Entered Review Mode: {item.review}]")
                assistant_responses.append(
                    ModelResponse(parts=[TextPart(content=f"Entered review mode: {item.review}")])
                )

            case ThreadItemExitedReviewMode():
                assistant_display_parts.append(f"[Exited Review Mode: {item.review}]")
                assistant_responses.append(
                    ModelResponse(parts=[TextPart(content=f"Exited review mode: {item.review}")])
                )

            case ThreadItemCollabAgentToolCall():
                status = item.agent_state.status if item.agent_state else "unknown"
                display = f"[Collab Agent: {item.tool}] {item.receiver_thread_id or ''} ({status})"
                assistant_display_parts.append(display)
                collab_args: dict[str, Any] = {
                    "tool": item.tool,
                    "sender_thread_id": item.sender_thread_id,
                }
                if item.receiver_thread_id:
                    collab_args["receiver_thread_id"] = item.receiver_thread_id
                if item.prompt:
                    collab_args["prompt"] = item.prompt
                collab_call = BuiltinToolCallPart(
                    tool_name="collab_agent", args=collab_args, tool_call_id=item.id
                )
                collab_ret = ToolReturnPart(
                    tool_name="collab_agent", content=f"Status: {status}", tool_call_id=item.id
                )
                assistant_responses.append(ModelResponse(parts=[collab_call]))
                assistant_responses.append(ModelRequest(parts=[collab_ret]))

    # Validate user content exists
    if not user_content:
        return []  # Skip turns with no user content
    result: list[ChatMessage[list[UserContent]]] = []
    user_msg: ChatMessage[list[UserContent]] = ChatMessage(
        content=user_content,
        role="user",
        message_id=f"{turn.id}-user",
        messages=[ModelRequest(parts=[UserPromptPart(content=user_content)])],
    )
    result.append(user_msg)

    # Create assistant message only if there are assistant responses
    if assistant_responses:
        display_text = "\n\n".join(assistant_display_parts) if assistant_display_parts else ""
        content: list[UserContent] = [display_text] if display_text else []
        assistant_msg: ChatMessage[list[UserContent]] = ChatMessage(
            content=content,
            role="assistant",
            message_id=f"{turn.id}-assistant",
            messages=assistant_responses,
        )
        result.append(assistant_msg)

    return result


def turns_to_chat_messages(turns: list[Turn]) -> list[ChatMessage[list[UserContent]]]:
    """Convert Codex turns to ChatMessage list for session loading.

    Each turn produces one or two ChatMessages:
    - User message with content as list[UserContent] (proper types for images etc.)
    - Assistant message (if present) with content as display text, plus messages field
      containing the full ModelMessage structure. Each ThreadItem becomes one
      "conversational beat" (one ModelResponse in the messages list).

    Handles incomplete/interrupted turns that may only have user content.

    Args:
        turns: List of Turn objects from Codex thread

    Returns:
        List of ChatMessages with proper content types and model messages
    """
    return [msg for turn in turns for msg in _turn_to_chat_messages(turn)]
