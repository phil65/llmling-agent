"""Mode definitions for agent behavior configuration.

Modes represent switchable behaviors/configurations that agents can expose
to clients. Each agent type can define its own mode categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModeInfo:
    """Information about a single mode option.

    Represents one selectable option within a mode category.
    """

    id: str
    """Unique identifier for this mode."""

    name: str
    """Human-readable display name."""

    description: str = ""
    """Optional description of what this mode does."""

    category_id: str = ""
    """ID of the category this mode belongs to."""


@dataclass
class ModeCategory:
    """A category of modes that can be switched.

    Represents a group of mutually exclusive modes. In the future,
    ACP may support multiple categories rendered as separate dropdowns.

    Examples:
        - Permissions: default, acceptEdits
        - Behavior: plan, code, architect
    """

    id: str
    """Unique identifier for this category."""

    name: str
    """Human-readable display name for the category."""

    available_modes: list[ModeInfo] = field(default_factory=list)
    """List of available modes in this category."""

    current_mode_id: str = ""
    """ID of the currently active mode."""
