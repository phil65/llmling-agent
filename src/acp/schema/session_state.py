"""Session state schema definitions."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from acp.schema.base import AnnotatedObject


class ModelInfo(AnnotatedObject):
    """**UNSTABLE**: This capability is not part of the spec yet.

    Information about a selectable model.
    """

    description: str | None = None
    """Optional description of the model."""

    model_id: str
    """Unique identifier for the model."""

    name: str
    """Human-readable name of the model."""


class SessionModelState(AnnotatedObject):
    """**UNSTABLE**: This capability is not part of the spec yet.

    The set of models and the one currently active.
    """

    available_models: Sequence[ModelInfo]
    """The set of models that the Agent can use."""

    current_model_id: str
    """The current model the Agent is using."""


class SessionMode(AnnotatedObject):
    """A mode the agent can operate in.

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """

    description: str | None = None
    """Optional description of the mode."""

    id: str
    """Unique identifier for the mode."""

    name: str
    """Human-readable name of the mode."""


class SessionModeState(AnnotatedObject):
    """The set of modes and the one currently active."""

    available_modes: Sequence[SessionMode]
    """The set of modes that the Agent can operate in."""

    current_mode_id: str
    """The current mode the Agent is in."""


class SessionInfo(AnnotatedObject):
    """**UNSTABLE**: This capability is not part of the spec yet.

    Information about a session returned by session/list.
    """

    cwd: str
    """The working directory for this session. Must be an absolute path."""

    session_id: str
    """Unique identifier for the session."""

    title: str | None = None
    """Human-readable title for the session."""

    updated_at: str | None = None
    """ISO 8601 timestamp of last activity."""
