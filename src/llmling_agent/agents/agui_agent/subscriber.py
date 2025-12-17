"""Subscriber system for AG-UI events.

Provides hooks for reacting to AG-UI events without subclassing AGUIAgent.
Inspired by the TypeScript SDK's AgentSubscriber pattern.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from ag_ui.core import (
        BaseEvent,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        StepFinishedEvent,
        StepStartedEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ThinkingTextMessageContentEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
    )

    from llmling_agent.agents.agui_agent.session_state import AGUISessionState


logger = get_logger(__name__)


# Type aliases for callback signatures
EventCallback = Callable[["BaseEvent", "AGUISessionState"], Awaitable[None] | None]
TextStartCallback = Callable[["TextMessageStartEvent", "AGUISessionState"], Awaitable[None] | None]
TextContentCallback = Callable[
    ["TextMessageContentEvent", str, "AGUISessionState"], Awaitable[None] | None
]
TextEndCallback = Callable[["TextMessageEndEvent", str, "AGUISessionState"], Awaitable[None] | None]
ToolStartCallback = Callable[["ToolCallStartEvent", "AGUISessionState"], Awaitable[None] | None]
ToolArgsCallback = Callable[
    ["ToolCallArgsEvent", str, dict[str, Any], "AGUISessionState"], Awaitable[None] | None
]
ToolEndCallback = Callable[
    ["ToolCallEndEvent", str, dict[str, Any], "AGUISessionState"], Awaitable[None] | None
]
ToolResultCallback = Callable[["ToolCallResultEvent", "AGUISessionState"], Awaitable[None] | None]
StateSnapshotCallback = Callable[[dict[str, Any], "AGUISessionState"], Awaitable[None] | None]
StateDeltaCallback = Callable[[list[Any], "AGUISessionState"], Awaitable[None] | None]
ThinkingCallback = Callable[
    ["ThinkingTextMessageContentEvent", str, "AGUISessionState"], Awaitable[None] | None
]
RunCallback = Callable[["AGUISessionState"], Awaitable[None] | None]
ErrorCallback = Callable[["RunErrorEvent", "AGUISessionState"], Awaitable[None] | None]


@runtime_checkable
class AGUISubscriberProtocol(Protocol):
    """Protocol for AG-UI event subscribers."""

    async def on_event(self, event: BaseEvent, state: AGUISessionState) -> None:
        """Called for every event."""
        ...


@dataclass
class AGUISubscriber:
    """Subscriber for AG-UI events with typed callbacks.

    Provides hooks for various AG-UI events. Callbacks can be sync or async.
    All callbacks receive the current session state for context.

    Example:
        ```python
        subscriber = AGUISubscriber(
            on_text_content=lambda event, buffer, state: print(f"Text: {event.delta}"),
            on_tool_call_end=lambda event, name, args, state: print(f"Tool: {name}({args})"),
        )

        agent = AGUIAgent(endpoint="...")
        unsubscribe = agent.subscribe(subscriber)

        # Later...
        unsubscribe()
        ```
    """

    # General event hook (called for all events)
    on_event: EventCallback | None = None

    # Text message hooks
    on_text_start: TextStartCallback | None = None
    on_text_content: TextContentCallback | None = None  # Receives accumulated buffer
    on_text_end: TextEndCallback | None = None  # Receives final text

    # Tool call hooks
    on_tool_call_start: ToolStartCallback | None = None
    on_tool_call_args: ToolArgsCallback | None = None  # Receives buffer + partial args
    on_tool_call_end: ToolEndCallback | None = None  # Receives name + final args
    on_tool_call_result: ToolResultCallback | None = None

    # State hooks
    on_state_snapshot: StateSnapshotCallback | None = None
    on_state_delta: StateDeltaCallback | None = None

    # Thinking/reasoning hooks
    on_thinking_start: (
        Callable[[ThinkingStartEvent, AGUISessionState], Awaitable[None] | None] | None
    ) = None
    on_thinking_content: ThinkingCallback | None = None  # Receives accumulated buffer
    on_thinking_end: (
        Callable[[ThinkingEndEvent, str, AGUISessionState], Awaitable[None] | None] | None
    ) = None

    # Lifecycle hooks
    on_run_started: Callable[[RunStartedEvent, AGUISessionState], Awaitable[None] | None] | None = (
        None
    )
    on_run_finished: (
        Callable[[RunFinishedEvent, AGUISessionState], Awaitable[None] | None] | None
    ) = None
    on_run_error: ErrorCallback | None = None
    on_step_started: (
        Callable[[StepStartedEvent, AGUISessionState], Awaitable[None] | None] | None
    ) = None
    on_step_finished: (
        Callable[[StepFinishedEvent, AGUISessionState], Awaitable[None] | None] | None
    ) = None


@dataclass
class SubscriberManager:
    """Manages multiple subscribers and dispatches events to them."""

    subscribers: list[AGUISubscriber] = field(default_factory=list)
    _text_buffers: dict[str, str] = field(default_factory=dict)
    _tool_buffers: dict[str, str] = field(default_factory=dict)
    _tool_names: dict[str, str] = field(default_factory=dict)
    _thinking_buffer: str = ""

    def add(self, subscriber: AGUISubscriber) -> Callable[[], None]:
        """Add a subscriber.

        Args:
            subscriber: Subscriber to add

        Returns:
            Unsubscribe function
        """
        self.subscribers.append(subscriber)
        return lambda: self.subscribers.remove(subscriber)

    def clear(self) -> None:
        """Remove all subscribers."""
        self.subscribers.clear()

    def reset_buffers(self) -> None:
        """Reset all internal buffers."""
        self._text_buffers.clear()
        self._tool_buffers.clear()
        self._tool_names.clear()
        self._thinking_buffer = ""

    async def dispatch(self, event: BaseEvent, state: AGUISessionState) -> None:  # noqa: PLR0915
        """Dispatch event to all subscribers.

        Args:
            event: AG-UI event
            state: Current session state
        """
        from ag_ui.core import (
            RunErrorEvent,
            RunFinishedEvent,
            RunStartedEvent,
            StateDeltaEvent,
            StateSnapshotEvent,
            StepFinishedEvent,
            StepStartedEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
            ThinkingEndEvent,
            ThinkingStartEvent,
            ThinkingTextMessageContentEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )
        import anyenv

        for subscriber in self.subscribers:
            try:
                # Always call on_event if defined
                if subscriber.on_event:
                    await _maybe_await(subscriber.on_event(event, state))

                # Dispatch to specific hooks based on event type
                match event:
                    # Text messages
                    case TextMessageStartEvent() as e:
                        self._text_buffers[e.message_id] = ""
                        if subscriber.on_text_start:
                            await _maybe_await(subscriber.on_text_start(e, state))

                    case TextMessageContentEvent() as e:
                        self._text_buffers[e.message_id] = (
                            self._text_buffers.get(e.message_id, "") + e.delta
                        )
                        if subscriber.on_text_content:
                            buffer = self._text_buffers[e.message_id]
                            await _maybe_await(subscriber.on_text_content(e, buffer, state))

                    case TextMessageEndEvent() as e:
                        if subscriber.on_text_end:
                            buffer = self._text_buffers.get(e.message_id, "")
                            await _maybe_await(subscriber.on_text_end(e, buffer, state))
                        self._text_buffers.pop(e.message_id, None)

                    # Tool calls
                    case ToolCallStartEvent() as e:
                        self._tool_buffers[e.tool_call_id] = ""
                        self._tool_names[e.tool_call_id] = e.tool_call_name
                        if subscriber.on_tool_call_start:
                            await _maybe_await(subscriber.on_tool_call_start(e, state))

                    case ToolCallArgsEvent() as e:
                        self._tool_buffers[e.tool_call_id] = (
                            self._tool_buffers.get(e.tool_call_id, "") + e.delta
                        )
                        if subscriber.on_tool_call_args:
                            buffer = self._tool_buffers[e.tool_call_id]
                            partial_args = _try_parse_partial_json(buffer)
                            await _maybe_await(
                                subscriber.on_tool_call_args(e, buffer, partial_args, state)
                            )

                    case ToolCallEndEvent() as e:
                        if subscriber.on_tool_call_end:
                            buffer = self._tool_buffers.get(e.tool_call_id, "")
                            name = self._tool_names.get(e.tool_call_id, "")
                            try:
                                args = anyenv.load_json(buffer) if buffer else {}
                            except anyenv.JsonLoadError:
                                args = {"raw": buffer}
                            await _maybe_await(subscriber.on_tool_call_end(e, name, args, state))
                        self._tool_buffers.pop(e.tool_call_id, None)
                        self._tool_names.pop(e.tool_call_id, None)

                    case ToolCallResultEvent() as e:
                        if subscriber.on_tool_call_result:
                            await _maybe_await(subscriber.on_tool_call_result(e, state))

                    # State
                    case StateSnapshotEvent() as e:
                        if subscriber.on_state_snapshot:
                            await _maybe_await(subscriber.on_state_snapshot(e.snapshot, state))

                    case StateDeltaEvent() as e:
                        if subscriber.on_state_delta:
                            await _maybe_await(subscriber.on_state_delta(e.delta, state))

                    # Thinking
                    case ThinkingStartEvent() as e:
                        self._thinking_buffer = ""
                        if subscriber.on_thinking_start:
                            await _maybe_await(subscriber.on_thinking_start(e, state))

                    case ThinkingTextMessageContentEvent() as e:
                        self._thinking_buffer += e.delta
                        if subscriber.on_thinking_content:
                            await _maybe_await(
                                subscriber.on_thinking_content(e, self._thinking_buffer, state)
                            )

                    case ThinkingEndEvent() as e:
                        if subscriber.on_thinking_end:
                            await _maybe_await(
                                subscriber.on_thinking_end(e, self._thinking_buffer, state)
                            )
                        self._thinking_buffer = ""

                    # Lifecycle
                    case RunStartedEvent() as e:
                        if subscriber.on_run_started:
                            await _maybe_await(subscriber.on_run_started(e, state))

                    case RunFinishedEvent() as e:
                        if subscriber.on_run_finished:
                            await _maybe_await(subscriber.on_run_finished(e, state))

                    case RunErrorEvent() as e:
                        if subscriber.on_run_error:
                            await _maybe_await(subscriber.on_run_error(e, state))

                    case StepStartedEvent() as e:
                        if subscriber.on_step_started:
                            await _maybe_await(subscriber.on_step_started(e, state))

                    case StepFinishedEvent() as e:
                        if subscriber.on_step_finished:
                            await _maybe_await(subscriber.on_step_finished(e, state))

            except Exception:
                logger.exception("Subscriber error", subscriber=subscriber)


async def _maybe_await(result: Any) -> None:
    """Await result if it's awaitable."""
    import inspect

    if inspect.isawaitable(result):
        await result


def _try_parse_partial_json(buffer: str) -> dict[str, Any]:
    """Try to parse partial/incomplete JSON.

    Uses simple heuristics to complete truncated JSON for preview purposes.

    Args:
        buffer: Potentially incomplete JSON string

    Returns:
        Parsed dict or empty dict if parsing fails
    """
    if not buffer:
        return {}

    # Try direct parse first
    try:
        import anyenv

        return anyenv.load_json(buffer)
    except Exception:  # noqa: BLE001
        pass

    # Try to complete truncated JSON by balancing braces/brackets
    try:
        import anyenv

        completed = buffer
        open_braces = buffer.count("{") - buffer.count("}")
        open_brackets = buffer.count("[") - buffer.count("]")

        # Add closing characters
        completed += "]" * max(0, open_brackets)
        completed += "}" * max(0, open_braces)

        return anyenv.load_json(completed)
    except Exception:  # noqa: BLE001
        return {}
