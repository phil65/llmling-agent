"""ACP Agent - MessageNode wrapping an external ACP subprocess."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Any

from agentpool.log import get_logger


if TYPE_CHECKING:
    from acp.schema import (
        AvailableCommandsUpdate,
        SessionConfigOption,
        SessionModelState,
        SessionModeState,
    )
    from agentpool.agents.events import RichAgentStreamEvent

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
    """Current model ID from session state (legacy)."""

    models: SessionModelState | None = None
    """Full model state including available models (legacy)."""

    modes: SessionModeState | None = None
    """Full mode state including available modes (legacy)."""

    current_mode_id: str | None = None
    """Current mode ID (legacy)."""

    config_options: list[SessionConfigOption] = dataclass_field(default_factory=list)
    """Unified session config options (replaces modes/models in newer ACP versions)."""

    available_commands: AvailableCommandsUpdate | None = None
    """Available commands from the agent."""

    def clear(self) -> None:
        self.events.clear()
        # Note: Don't clear current_model_id, models, config_options - those persist
