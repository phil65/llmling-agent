"""Event sources for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from evented.event_data import EventData


if TYPE_CHECKING:
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.talk.talk import Talk
    from llmling_agent_config.events import ConnectionEventType


ChangeType = Literal["added", "modified", "deleted"]


class ConnectionEventData[TTransmittedData](EventData):
    """Event from connection activity."""

    connection_name: str
    """Name of the connection which fired an event."""

    connection: Talk[TTransmittedData]
    """The connection which fired the event."""

    event_type: ConnectionEventType
    """Type of event that occurred."""

    message: ChatMessage[TTransmittedData] | None = None
    """The message at the stage of the event."""

    def to_prompt(self) -> str:
        """Convert event to agent prompt."""
        base = f"Connection {self.connection_name!r} event: {self.event_type}"
        if self.message:
            return f"{base}\nMessage content: {self.message.content}"
        return base
