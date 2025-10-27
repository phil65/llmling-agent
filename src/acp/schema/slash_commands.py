"""Slash command schema definitions."""

from __future__ import annotations

from typing import Any

from pydantic import RootModel

from acp.base import Schema


class CommandInputHint(Schema):
    """All text that was typed after the command name is provided as input."""

    hint: str
    """A hint to display when the input hasn't been provided yet."""


class AvailableCommandInput(RootModel[CommandInputHint]):
    """A container for the input specification for a command."""

    root: CommandInputHint
    """The input specification for a command."""


class AvailableCommand(Schema):
    """Information about a command."""

    field_meta: Any | None = None
    """Extension point for implementations."""

    description: str
    """Human-readable description of what the command does."""

    input: AvailableCommandInput | None = None
    """Input for the command if required."""

    name: str
    """Command name (e.g., `create_plan`, `research_codebase`)."""
