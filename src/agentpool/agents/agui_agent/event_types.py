"""Extended AG-UI event types.

The upstream ag_ui.core.Event union is missing thinking events.
This module provides a complete Event type until the SDK is fixed.

TODO: Remove this workaround once upstream SDK is fixed.
See: https://github.com/ag-ui-protocol/ag-ui/pull/753
"""

from __future__ import annotations

from typing import Annotated

from ag_ui.core import (
    ActivityDeltaEvent,
    ActivitySnapshotEvent,
    CustomEvent,
    MessagesSnapshotEvent,
    RawEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from pydantic import Field


# Complete Event union including thinking events (missing from upstream SDK)
Event = Annotated[
    # Text message events
    TextMessageStartEvent
    | TextMessageContentEvent
    | TextMessageEndEvent
    | TextMessageChunkEvent
    # Thinking events (missing from upstream Event union)
    | ThinkingStartEvent
    | ThinkingEndEvent
    | ThinkingTextMessageStartEvent
    | ThinkingTextMessageContentEvent
    | ThinkingTextMessageEndEvent
    # Tool call events
    | ToolCallStartEvent
    | ToolCallArgsEvent
    | ToolCallEndEvent
    | ToolCallChunkEvent
    | ToolCallResultEvent
    # State events
    | StateSnapshotEvent
    | StateDeltaEvent
    | MessagesSnapshotEvent
    # Activity events
    | ActivitySnapshotEvent
    | ActivityDeltaEvent
    # Lifecycle events
    | RunStartedEvent
    | RunFinishedEvent
    | RunErrorEvent
    | StepStartedEvent
    | StepFinishedEvent
    # Special events
    | RawEvent
    | CustomEvent,
    Field(discriminator="type"),
]

__all__ = ["Event"]
