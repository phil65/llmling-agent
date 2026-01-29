from typing import cast
from unittest.mock import MagicMock

from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, ThinkingPart, ThinkingPartDelta

from agentpool_server.opencode_server.models import PartUpdatedEvent
from agentpool_server.opencode_server.models.events import PartUpdatedEventProperties
from agentpool_server.opencode_server.models.parts import ReasoningPart
from agentpool_server.opencode_server.stream_adapter import OpenCodeStreamAdapter


def test_thinking_events_create_reasoning_part():
    """Verify ThinkingPart/ThinkingPartDelta events create ReasoningPart."""
    # Create a mock MessageWithParts
    mock_msg = MagicMock()
    mock_msg.parts = []

    adapter = OpenCodeStreamAdapter(
        session_id="test-session",
        assistant_msg_id="msg-1",
        assistant_msg=mock_msg,
        working_dir=".",
    )

    # Use the adapter's _handle_event method directly
    events = list(
        adapter._handle_event(PartStartEvent(index=0, part=ThinkingPart(content="Thinking...")))
    )
    events.extend(
        list(
            adapter._handle_event(
                PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" more..."))
            )
        )
    )

    # Assert reasoning part was created
    # Based on models/events.py, PartUpdatedEvent has properties.part
    reasoning_events = []
    for e in events:
        if isinstance(e, PartUpdatedEvent):
            props = e.properties
            if isinstance(props, PartUpdatedEventProperties) and isinstance(
                props.part, ReasoningPart
            ):
                reasoning_events.append(e)

    assert len(reasoning_events) >= 1, "ReasoningPart should be created from thinking events"
    # Cast to narrow type since we've already checked it's a ReasoningPart
    first_part = cast(ReasoningPart, reasoning_events[0].properties.part)
    last_part = cast(ReasoningPart, reasoning_events[-1].properties.part)
    assert "Thinking..." in first_part.text
    assert " more..." in last_part.text
