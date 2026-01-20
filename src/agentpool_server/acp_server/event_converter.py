"""Convert agent stream events to ACP notifications.

This module provides a stateful converter that transforms agent stream events
into ACP session update objects. The converter tracks tool call state but does
not perform any I/O - it yields notification objects that can be emitted
by the caller.

This separation enables easy testing without mocks.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
    ToolReturnPart,
)

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ContentToolCallContent,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)
from acp.utils import generate_tool_title, infer_tool_kind, to_acp_content_blocks
from agentpool.agents.events import (
    CompactionEvent,
    DiffContentItem,
    FileContentItem,
    LocationContentItem,
    PlanUpdateEvent,
    RunErrorEvent,
    StreamCompleteEvent,
    SubAgentEvent,
    TerminalContentItem,
    TextContentItem,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from agentpool.log import get_logger
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from acp.schema.tool_call import ToolCallContent, ToolCallKind
    from agentpool.agents.events import RichAgentStreamEvent

logger = get_logger(__name__)


# Type alias for all session updates the converter can yield
ACPSessionUpdate = (
    AgentMessageChunk | AgentThoughtChunk | ToolCallStart | ToolCallProgress | AgentPlanUpdate
)


# ============================================================================
# Internal tool state tracking
# ============================================================================


@dataclass
class _ToolState:
    """Internal state for a single tool call."""

    tool_call_id: str
    tool_name: str
    title: str
    kind: ToolCallKind
    raw_input: dict[str, Any]
    started: bool = False
    has_content: bool = False


# ============================================================================
# Event Converter
# ============================================================================


@dataclass
class ACPEventConverter:
    """Converts agent stream events to ACP session updates.

    Stateful converter that tracks tool calls and subagent content,
    yielding ACP schema objects without performing I/O.

    Example:
        ```python
        converter = ACPEventConverter()
        async for event in agent.run_stream(...):
            async for update in converter.convert(event):
                await client.session_update(SessionNotification(session_id=sid, update=update))
        ```
    """

    subagent_display_mode: Literal["inline", "tool_box"] = "tool_box"
    """How to display subagent output."""

    # Internal state
    _tool_states: dict[str, _ToolState] = field(default_factory=dict)
    """Active tool call states."""

    _current_tool_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Current tool inputs by tool_call_id."""

    _subagent_headers: set[str] = field(default_factory=set)
    """Track which subagent headers have been sent (for inline mode)."""

    _subagent_content: dict[str, list[str]] = field(default_factory=dict)
    """Accumulated content per subagent (for tool_box mode)."""

    def reset(self) -> None:
        """Reset converter state for a new run."""
        self._tool_states.clear()
        self._current_tool_inputs.clear()
        self._subagent_headers.clear()
        self._subagent_content.clear()

    async def cancel_pending_tools(self) -> AsyncIterator[ToolCallProgress]:
        """Cancel all pending tool calls.

        Yields ToolCallProgress notifications with status="completed" for all
        tool calls that were started but not completed. This should be called
        when the stream is interrupted to properly clean up client-side state.

        Note:
            Uses status="completed" since ACP doesn't have a "cancelled" status.
            This signals to the client that we're done with these tool calls.

        Yields:
            ToolCallProgress notifications for each pending tool call
        """
        for tool_call_id, state in list(self._tool_states.items()):
            if state.started:
                yield ToolCallProgress(
                    tool_call_id=tool_call_id,
                    status="completed",
                )
        # Clean up all state
        self.reset()

    def _get_or_create_tool_state(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> _ToolState:
        """Get existing tool state or create a new one."""
        if tool_call_id not in self._tool_states:
            self._tool_states[tool_call_id] = _ToolState(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=generate_tool_title(tool_name, tool_input),
                kind=infer_tool_kind(tool_name),
                raw_input=tool_input,
            )
        return self._tool_states[tool_call_id]

    def _cleanup_tool_state(self, tool_call_id: str) -> None:
        """Remove tool state after completion."""
        self._tool_states.pop(tool_call_id, None)
        self._current_tool_inputs.pop(tool_call_id, None)

    async def convert(  # noqa: PLR0915
        self, event: RichAgentStreamEvent[Any]
    ) -> AsyncIterator[ACPSessionUpdate]:
        """Convert an agent event to zero or more ACP session updates."""
        from acp.schema import (
            FileEditToolCallContent,
            PlanEntry as ACPPlanEntry,
            TerminalToolCallContent,
        )
        from agentpool_server.acp_server.syntax_detection import format_zed_code_block

        match event:
            # Text output
            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                yield AgentMessageChunk.text(text=delta)

            # Thinking/reasoning
            case (
                PartStartEvent(part=ThinkingPart(content=delta))
                | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
            ):
                yield AgentThoughtChunk.text(text=delta or "\n")

            # Builtin tool call started (e.g., WebSearchTool, CodeExecutionTool)
            case PartStartEvent(part=BuiltinToolCallPart() as part):
                tool_call_id = part.tool_call_id
                tool_input = safe_args_as_dict(part, default={})
                self._current_tool_inputs[tool_call_id] = tool_input
                state = self._get_or_create_tool_state(tool_call_id, part.tool_name, tool_input)
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=state.title,
                        kind=state.kind,
                        raw_input=state.raw_input,
                        status="pending",
                    )

            # Builtin tool completed
            case PartStartEvent(part=BuiltinToolReturnPart() as part):
                tool_call_id = part.tool_call_id
                tool_state = self._tool_states.get(tool_call_id)
                final_output = part.content

                if tool_state and tool_state.has_content:
                    yield ToolCallProgress(
                        tool_call_id=tool_call_id,
                        status="completed",
                        raw_output=final_output,
                    )
                else:
                    converted = to_acp_content_blocks(final_output)
                    content = [ContentToolCallContent(content=block) for block in converted]
                    yield ToolCallProgress(
                        tool_call_id=tool_call_id,
                        status="completed",
                        raw_output=final_output,
                        content=content,
                    )
                self._cleanup_tool_state(tool_call_id)

            case PartStartEvent(part=part):
                logger.debug("Received unhandled PartStartEvent", part=part)

            # Tool call streaming delta
            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                if delta_part := delta.as_part():
                    tool_call_id = delta_part.tool_call_id
                    try:
                        tool_input = delta_part.args_as_dict()
                    except ValueError:
                        pass  # Args still streaming, not valid JSON yet
                    else:
                        self._current_tool_inputs[tool_call_id] = tool_input
                        state = self._get_or_create_tool_state(
                            tool_call_id, delta_part.tool_name, tool_input
                        )
                        if not state.started:
                            state.started = True
                            yield ToolCallStart(
                                tool_call_id=tool_call_id,
                                title=state.title,
                                kind=state.kind,
                                raw_input=state.raw_input,
                                status="pending",
                            )

            # Function tool call started
            case FunctionToolCallEvent(part=part):
                tool_call_id = part.tool_call_id
                tool_input = safe_args_as_dict(part, default={})
                self._current_tool_inputs[tool_call_id] = tool_input
                state = self._get_or_create_tool_state(tool_call_id, part.tool_name, tool_input)
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=state.title,
                        kind=state.kind,
                        raw_input=state.raw_input,
                        status="pending",
                    )

            # Tool completed successfully
            case FunctionToolResultEvent(
                result=ToolReturnPart(content=content) as result,
                tool_call_id=tool_call_id,
            ):
                # Handle async generator content
                if isinstance(content, AsyncGenerator):
                    full_content = ""
                    async for chunk in content:
                        full_content += str(chunk)
                        if tool_call_id in self._tool_states:
                            yield ToolCallProgress(
                                tool_call_id=tool_call_id,
                                status="in_progress",
                                raw_output=chunk,
                            )
                    result.content = full_content
                    final_output = full_content
                else:
                    final_output = result.content

                tool_state = self._tool_states.get(tool_call_id)
                if tool_state and tool_state.has_content:
                    yield ToolCallProgress(
                        tool_call_id=tool_call_id,
                        status="completed",
                        raw_output=final_output,
                    )
                else:
                    converted = to_acp_content_blocks(final_output)
                    content_items = [ContentToolCallContent(content=block) for block in converted]
                    yield ToolCallProgress(
                        tool_call_id=tool_call_id,
                        status="completed",
                        raw_output=final_output,
                        content=content_items,
                    )
                self._cleanup_tool_state(tool_call_id)

            # Tool failed with retry
            case FunctionToolResultEvent(
                result=RetryPromptPart() as result,
                tool_call_id=tool_call_id,
            ):
                error_message = result.model_response()
                yield ToolCallProgress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    content=[ContentToolCallContent.text(text=f"Error: {error_message}")],
                )
                self._cleanup_tool_state(tool_call_id)

            # Tool emits its own start event
            case ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=title,
                kind=kind,
                locations=loc_items,
                raw_input=raw_input,
            ):
                state = self._get_or_create_tool_state(tool_call_id, tool_name, raw_input or {})
                acp_locations = [ToolCallLocation(path=i.path, line=i.line) for i in loc_items]
                # If not started, send start notification
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=title,
                        kind=kind,
                        raw_input=raw_input,
                        locations=acp_locations or None,
                        status="pending",
                    )
                else:
                    # Send update with tool-provided details
                    yield ToolCallProgress(
                        tool_call_id=tool_call_id,
                        title=title,
                        kind=kind,
                        locations=acp_locations or None,
                    )

            # Tool progress event - create state if needed (tool may emit progress before SDK event)
            case ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                title=title,
                # status=status,
                items=items,
                progress=progress,
                total=total,
                message=message,
            ) if tool_call_id:
                # Get or create state - handles race where tool emits before SDK event
                state = self._get_or_create_tool_state(tool_call_id, "unknown", {})
                # Emit start if this is the first event for this tool call
                if not state.started:
                    state.started = True
                    yield ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=title or state.title,
                        kind=state.kind,
                        raw_input=state.raw_input,
                        status="pending",
                    )
                acp_content: list[ToolCallContent] = []
                progress_locations: list[ToolCallLocation] = []
                for item in items:
                    match item:
                        case TerminalContentItem(terminal_id=tid):
                            acp_content.append(TerminalToolCallContent(terminal_id=tid))
                        case TextContentItem(text=text):
                            acp_content.append(ContentToolCallContent.text(text=text))
                        case FileContentItem(
                            content=file_content,
                            path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        ):
                            formatted = format_zed_code_block(
                                file_content, file_path, start_line, end_line
                            )
                            acp_content.append(ContentToolCallContent.text(text=formatted))
                            progress_locations.append(
                                ToolCallLocation(path=file_path, line=start_line or 0)
                            )
                        case DiffContentItem(path=diff_path, old_text=old, new_text=new):
                            acp_content.append(
                                FileEditToolCallContent(path=diff_path, old_text=old, new_text=new)
                            )
                            progress_locations.append(ToolCallLocation(path=diff_path))
                            state.has_content = True
                        case LocationContentItem(path=loc_path, line=loc_line):
                            location = ToolCallLocation(path=loc_path, line=loc_line)
                            progress_locations.append(location)

                # Build title: use provided title, or format MCP numeric progress
                effective_title = title
                if not effective_title and (progress is not None or message):
                    # MCP-style numeric progress - format into title
                    if progress is not None and total:
                        pct = int(progress / total * 100)
                        effective_title = f"{message} ({pct}%)" if message else f"Progress: {pct}%"
                    elif message:
                        effective_title = message

                # TODO: Progress events shouldn't control completion status.
                # The file_operation helper sets status="completed" on success, but that's
                # emitted mid-operation (before content display). Only FunctionToolResultEvent
                # should mark a tool as completed. For now, hardcode in_progress.
                yield ToolCallProgress(
                    tool_call_id=tool_call_id,
                    title=effective_title,
                    status="in_progress",
                    content=acp_content or None,
                    locations=progress_locations or None,
                )
                if acp_content:
                    state.has_content = True

            case FinalResultEvent():
                pass  # No notification needed

            case StreamCompleteEvent():
                pass  # No notification needed

            case PlanUpdateEvent(entries=entries):
                acp_entries = [
                    ACPPlanEntry(content=e.content, priority=e.priority, status=e.status)
                    for e in entries
                ]
                yield AgentPlanUpdate(entries=acp_entries)

            case CompactionEvent(trigger=trigger, phase=phase):
                if phase == "starting":
                    if trigger == "auto":
                        text = (
                            "\n\n---\n\n"
                            "📦 **Context compaction** triggered. "
                            "Summarizing conversation..."
                            "\n\n---\n\n"
                        )
                    else:
                        text = (
                            "\n\n---\n\n"
                            "📦 **Manual compaction** requested. "
                            "Summarizing conversation..."
                            "\n\n---\n\n"
                        )
                    yield AgentMessageChunk.text(text=text)

            case SubAgentEvent(
                source_name=source_name,
                source_type=source_type,
                event=inner_event,
                depth=depth,
            ):
                if self.subagent_display_mode == "tool_box":
                    async for update in self._convert_subagent_tool_box(
                        source_name, source_type, inner_event, depth
                    ):
                        yield update
                else:
                    async for update in self._convert_subagent_inline(
                        source_name, source_type, inner_event, depth
                    ):
                        yield update

            case RunErrorEvent(message=message, agent_name=agent_name):
                # Display error as agent text with formatting
                agent_prefix = f"[{agent_name}] " if agent_name else ""
                error_text = f"\n\n❌ **Error**: {agent_prefix}{message}\n\n"
                yield AgentMessageChunk.text(text=error_text)

            case _:
                logger.debug("Unhandled event", event_type=type(event).__name__)

    async def _convert_subagent_inline(
        self,
        source_name: str,
        source_type: Literal["agent", "team_parallel", "team_sequential"],
        inner_event: RichAgentStreamEvent[Any],
        depth: int,
    ) -> AsyncIterator[ACPSessionUpdate]:
        """Convert subagent event to inline text notifications."""
        indent = "  " * depth
        icon = "🤖" if source_type == "agent" else "👥"

        match inner_event:
            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                header_key = f"{source_name}:{depth}"
                if header_key not in self._subagent_headers:
                    self._subagent_headers.add(header_key)
                    yield AgentMessageChunk.text(text=f"\n{indent}{icon} **{source_name}**: ")
                yield AgentMessageChunk.text(text=delta)

            case (
                PartStartEvent(part=ThinkingPart(content=delta))
                | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
            ):
                yield AgentThoughtChunk.text(text=f"{indent}[{source_name}] {delta or ''}")

            case FunctionToolCallEvent(part=part):
                text = f"\n{indent}🔧 [{source_name}] Using tool: {part.tool_name}\n"
                yield AgentMessageChunk.text(text=text)

            case FunctionToolResultEvent(
                result=ToolReturnPart(content=content, tool_name=tool_name),
            ):
                result_str = str(content)
                if len(result_str) > 200:  # noqa: PLR2004
                    result_str = result_str[:200] + "..."
                text = f"{indent}✅ [{source_name}] {tool_name}: {result_str}\n"
                yield AgentMessageChunk.text(text=text)

            case FunctionToolResultEvent(result=RetryPromptPart(tool_name=tool_name) as result):
                error_msg = result.model_response()
                text = f"{indent}❌ [{source_name}] {tool_name}: {error_msg}\n"
                yield AgentMessageChunk.text(text=text)

            case StreamCompleteEvent():
                header_key = f"{source_name}:{depth}"
                self._subagent_headers.discard(header_key)
                yield AgentMessageChunk.text(text=f"\n{indent}---\n")

            case _:
                pass

    async def _convert_subagent_tool_box(
        self,
        source_name: str,
        source_type: Literal["agent", "team_parallel", "team_sequential"],
        inner_event: RichAgentStreamEvent[Any],
        depth: int,
    ) -> AsyncIterator[ACPSessionUpdate]:
        """Convert subagent event to tool box notifications."""
        state_key = f"subagent:{source_name}:{depth}"
        icon = "🤖" if source_type == "agent" else "👥"

        if state_key not in self._subagent_content:
            self._subagent_content[state_key] = []

        accumulated = self._subagent_content[state_key]

        match inner_event:
            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                accumulated.append(delta)
                async for n in self._emit_subagent_progress(state_key, f"{icon} {source_name}"):
                    yield n

            case (
                PartStartEvent(part=ThinkingPart(content=delta))
                | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
            ):
                if delta:
                    accumulated.append(f"💭 {delta}")
                    async for n in self._emit_subagent_progress(state_key, f"{icon} {source_name}"):
                        yield n

            case FunctionToolCallEvent(part=part):
                accumulated.append(f"\n🔧 Using tool: {part.tool_name}\n")
                async for n in self._emit_subagent_progress(state_key, f"{icon} {source_name}"):
                    yield n

            case FunctionToolResultEvent(
                result=ToolReturnPart(content=content, tool_name=tool_name),
            ):
                result_str = str(content)
                if len(result_str) > 200:  # noqa: PLR2004
                    result_str = result_str[:200] + "..."
                accumulated.append(f"✅ {tool_name}: {result_str}\n")
                async for n in self._emit_subagent_progress(state_key, f"{icon} {source_name}"):
                    yield n

            case FunctionToolResultEvent(result=RetryPromptPart(tool_name=tool_name) as result):
                error_msg = result.model_response()
                accumulated.append(f"❌ {tool_name}: {error_msg}\n")
                async for n in self._emit_subagent_progress(state_key, f"{icon} {source_name}"):
                    yield n

            case StreamCompleteEvent():
                # Complete the tool call
                if state_key in self._tool_states:
                    yield ToolCallProgress(tool_call_id=state_key, status="completed")
                    self._cleanup_tool_state(state_key)
                self._subagent_content.pop(state_key, None)

            case _:
                pass

    async def _emit_subagent_progress(
        self, state_key: str, title: str
    ) -> AsyncIterator[ACPSessionUpdate]:
        """Emit tool call notifications for subagent content."""
        accumulated = self._subagent_content.get(state_key, [])
        content_text = "".join(accumulated)

        if state_key not in self._tool_states:
            # Create state and emit start
            self._tool_states[state_key] = _ToolState(
                tool_call_id=state_key,
                tool_name="delegate",
                title=title,
                kind="other",
                raw_input={},
                started=True,
            )
            yield ToolCallStart(
                tool_call_id=state_key,
                title=title,
                kind="other",
                raw_input={},
                status="pending",
            )

        # Emit progress update
        yield ToolCallProgress(
            tool_call_id=state_key,
            title=title,
            status="in_progress",
            content=[ContentToolCallContent.text(text=content_text)],
        )
