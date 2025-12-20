"""ACP Agent - MessageNode wrapping an external ACP subprocess."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from acp.schema import SessionModelState, SessionModeState
    from llmling_agent.agents.events import RichAgentStreamEvent

logger = get_logger(__name__)

PROTOCOL_VERSION = 1


@dataclass
class ACPSessionState:
    """Tracks state of an ACP session."""

    session_id: str
    """The session ID from the ACP server."""

    events: list[RichAgentStreamEvent[Any]] = dataclass_field(default_factory=list)
    """Queue of native events converted from ACP updates."""

    current_model_id: str | None = None
    """Current model ID from session state."""

    models: SessionModelState | None = None
    """Full model state including available models from nested ACP agent."""

    modes: SessionModeState | None = None
    """Full mode state including available modes from nested ACP agent."""

    current_mode_id: str | None = None
    """Current mode ID."""

    def clear(self) -> None:
        self.events.clear()
        # Note: Don't clear current_model_id or models - those persist across prompts
