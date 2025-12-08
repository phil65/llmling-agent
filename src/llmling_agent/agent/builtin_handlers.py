"""Built-in event handlers for simple console output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from llmling_agent.agent.events import (
    FileEditProgressEvent,
    ProcessExitEvent,
    ProcessStartEvent,
    RunErrorEvent,
    StreamCompleteEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import AgentStreamEvent, RunContext

    from llmling_agent.agent.events import (
        RichAgentStreamEvent,
    )
    from llmling_agent.common_types import BuiltinEventHandlerType, IndividualEventHandler


async def simple_print_handler(ctx: RunContext, event: AgentStreamEvent) -> None:
    """Simple event handler that prints text and basic tool information.

    Focus: Core text output and minimal tool notifications.
    Prints:
    - Text content (streaming)
    - Tool calls (name only)
    - Errors
    """
    match event:
        case (
            PartStartEvent(part=TextPart(content=delta))
            | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
        ):
            print(delta, end="", flush=True)

        case FunctionToolCallEvent(part=ToolCallPart(tool_name=tool_name)):
            print(f"\n🔧 {tool_name}", flush=True)

        case FunctionToolResultEvent(result=ToolReturnPart()):
            pass  # Silent completion

        case RunErrorEvent(message=message):
            print(f"\n❌ Error: {message}", flush=True)

        case StreamCompleteEvent():
            print()  # Final newline


async def detailed_print_handler(ctx: RunContext, event: RichAgentStreamEvent[Any]) -> None:
    """Detailed event handler with rich tool execution information.

    Focus: Comprehensive execution visibility.
    Prints:
    - Text content (streaming)
    - Thinking content
    - Tool calls with inputs
    - Tool results
    - Process execution
    - File operations
    - Errors
    """
    match event:
        case (
            PartStartEvent(part=TextPart(content=delta))
            | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
        ):
            print(delta, end="", flush=True)

        case (
            PartStartEvent(part=ThinkingPart(content=delta))
            | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
        ):
            if delta:
                print(f"\n💭 {delta}", end="", flush=True)

        case ToolCallStartEvent(tool_name=tool_name, title=title, tool_call_id=call_id):
            print(f"\n🔧 {tool_name}: {title} (#{call_id[:8]})", flush=True)

        case FunctionToolCallEvent(part=ToolCallPart(tool_name=tool_name, args=args)):
            args_str = str(args)
            if len(args_str) > 100:  # noqa: PLR2004
                args_str = args_str[:97] + "..."
            print(f"  📝 Input: {args_str}", flush=True)

        case FunctionToolResultEvent(result=ToolReturnPart(content=content, tool_name=tool_name)):
            result_str = str(content)
            if len(result_str) > 150:  # noqa: PLR2004
                result_str = result_str[:147] + "..."
            print(f"  ✅ {tool_name}: {result_str}", flush=True)

        case ProcessStartEvent(process_id=pid, command=command, success=success):
            if success:
                print(f"  🖥️  Started process: {command} (PID: {pid})", flush=True)
            else:
                print(f"  ❌ Failed to start: {command}", flush=True)

        case ProcessExitEvent(process_id=pid, exit_code=code, success=success):
            status = "✅" if success else "❌"
            print(f"  {status} Process {pid} exited: code {code}", flush=True)

        case FileEditProgressEvent(path=path, status=status):
            emoji = {"in_progress": "✏️", "completed": "✅", "failed": "❌"}.get(status, "📝")
            print(f"  {emoji} {status}: {path}", flush=True)

        case RunErrorEvent(message=message, code=code):
            error_info = f" [{code}]" if code else ""
            print(f"\n❌ Error{error_info}: {message}", flush=True)

        case StreamCompleteEvent():
            print()  # Final newline


def resolve_event_handlers(
    event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None,
) -> list[IndividualEventHandler]:
    """Resolve event handlers, converting builtin handler names to actual handlers."""
    if not event_handlers:
        return []
    builtin_map = {"simple": simple_print_handler, "detailed": detailed_print_handler}
    return [builtin_map[h] if isinstance(h, str) else h for h in event_handlers]
