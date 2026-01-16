"""Mode definitions for agent behavior configuration.

Modes represent switchable behaviors/configurations that agents can expose
to clients. Each agent type can define its own mode categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from agentpool.agents.base_agent import BaseAgent

# TypeVar for generic protocol - contravariant because it's used in method parameters
AgentT_contra = TypeVar("AgentT_contra", bound="BaseAgent", contravariant=True)


# Standard config option IDs aligned with ACP's SessionConfigOptionCategory.
# See: src/acp/schema/session_state.py
ModeCategoryId = Literal[
    "mode",  # Session mode / permissions / approval policy
    "model",  # Model selection
    "thought_level",  # Thinking/reasoning effort level
]


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

    category_id: ModeCategoryId | str = ""
    """ID of the category this mode belongs to."""


@dataclass
class ConfigOptionChanged:
    """Atomic state change for a config option.

    Emitted by agents when a config option value changes.
    The ACP layer converts this to ACP's ConfigOptionUpdate.
    """

    config_id: str
    """ID of the config option that changed (e.g., 'permissions', 'model')."""

    value_id: str
    """New value ID for this config option."""


@runtime_checkable
class ModeCategoryProtocol(Protocol[AgentT_contra]):
    """Protocol for mode categories that can apply themselves.

    Generic over agent type. Use ModeCategoryProtocol[CodexAgent] for
    Codex-specific categories, ModeCategoryProtocol[BaseAgent] for
    agent-agnostic ones.

    State (current mode) lives on the agent - categories just know how to
    read and write it via get_current() and apply().
    """

    @property
    def id(self) -> str:
        """Unique identifier for this category."""

    @property
    def name(self) -> str:
        """Human-readable display name for the category."""

    @property
    def available_modes(self) -> list[ModeInfo]:
        """List of available modes in this category."""

    @property
    def category(self) -> str | None:
        """Optional semantic category for UX purposes."""

    def get_current(self, agent: AgentT_contra) -> str:
        """Get the current mode ID from the agent."""

    async def apply(self, agent: AgentT_contra, mode_id: str) -> None:
        """Apply a mode to the agent."""


@dataclass
class ModeCategory:
    """Data container for mode category information.

    Used by get_modes() to expose available modes to clients.
    For mode-switching behavior, see ModeCategoryProtocol.

    Examples:
        - Permissions: default, acceptEdits
        - Behavior: plan, code, architect

    TODO: Consider adding an `is_static` field to indicate whether modes are
    known in advance (Codex, Claude Code) vs dynamic (ACP where modes can
    change based on model selection). This would help consumers like OpenCode
    decide whether to fetch modes once at startup or handle updates.
    """

    id: ModeCategoryId | str
    """Unique identifier for this category."""

    name: str
    """Human-readable display name for the category."""

    available_modes: list[ModeInfo] = field(default_factory=list)
    """List of available modes in this category."""

    current_mode_id: str = ""
    """ID of the currently active mode."""

    category: Literal["mode", "model", "thought_level", "other"] | None = None
    """Optional semantic category for UX purposes (keyboard shortcuts, icons, placement).

    This helps clients distinguish common selector types:
    - 'mode': Session mode selector
    - 'model': Model selector
    - 'thought_level': Thought/reasoning level selector
    - 'other': Unknown/uncategorized

    MUST NOT be required for correctness. Clients should handle gracefully.
    """
