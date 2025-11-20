"""Agent pool management for collaboration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.talk import Talk


class MessageFlowTracker:
    """Class for tracking message flow in conversations."""

    def __init__(self) -> None:
        self.events: list[Talk.ConnectionProcessed] = []

    def track(self, event: Talk.ConnectionProcessed) -> None:
        self.events.append(event)

    def filter(self, message: ChatMessage[Any]) -> list[ChatMessage[Any]]:
        """Filter events for specific conversation."""
        return [
            e.message for e in self.events if e.message.conversation_id == message.conversation_id
        ]

    def visualize(self, message: ChatMessage[Any]) -> str:
        """Get flow visualization for specific conversation."""
        # Filter events for this conversation
        conv_events = [
            e for e in self.events if e.message.conversation_id == message.conversation_id
        ]
        lines = ["flowchart LR"]
        for event in conv_events:
            source = event.message.name
            for target in event.targets:
                lines.append(f"    {source}-->{target.name}")  # noqa: PERF401
        return "\n".join(lines)
