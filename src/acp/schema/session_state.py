"""Session state schema definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field

from acp.schema.base import AnnotatedObject


# Type aliases for config option identifiers
SessionConfigId = str
"""Unique identifier for a configuration option."""

SessionConfigValueId = str
"""Unique identifier for a possible value within a configuration option."""

SessionConfigGroupId = str
"""Unique identifier for a group of values within a configuration option."""


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


class SessionConfigSelectOption(AnnotatedObject):
    """A possible value for a configuration selector."""

    id: SessionConfigValueId
    """Unique identifier for this value."""

    label: str
    """Human-readable label for this value."""

    description: str | None = None
    """Optional description explaining this value."""


class SessionConfigSelectGroup(AnnotatedObject):
    """A group of possible values for a configuration selector."""

    id: SessionConfigGroupId
    """Unique identifier for this group."""

    label: str
    """Human-readable label for this group."""

    options: Sequence[SessionConfigSelectOption]
    """The options within this group."""


SessionConfigSelectOptions = Sequence[SessionConfigSelectOption | SessionConfigSelectGroup]
"""The possible values for a configuration selector, optionally organized into groups."""


class SessionConfigSelect(AnnotatedObject):
    """A configuration option that allows selecting a single value from a list.

    Similar to a dropdown/select UI element.
    """

    type: Literal["select"] = Field(default="select", init=False)
    """Discriminator for the config option type."""

    id: SessionConfigId
    """Unique identifier for this configuration option."""

    label: str
    """Human-readable label for this option."""

    description: str | None = None
    """Optional description explaining this option."""

    options: SessionConfigSelectOptions
    """The possible values for this option."""

    value: SessionConfigValueId
    """The currently selected value ID."""


SessionConfigOption = Annotated[
    SessionConfigSelect,
    Field(discriminator="type"),
]
"""A session configuration option.

Currently only supports select-type options, but designed for extensibility.
"""
