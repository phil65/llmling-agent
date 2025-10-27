"""Slash command schema definitions."""

from __future__ import annotations

from pydantic import RootModel

from acp.schema.base import AnnotatedObject, Schema


class CommandInputHint(Schema):
    """All text that was typed after the command name is provided as input."""

    hint: str
    """A hint to display when the input hasn't been provided yet."""


class AvailableCommandInput(RootModel[CommandInputHint]):
    """A container for the input specification for a command."""

    root: CommandInputHint
    """The input specification for a command."""


class AvailableCommand(AnnotatedObject):
    """Information about a command."""

    description: str
    """Human-readable description of what the command does."""

    input: AvailableCommandInput | None = None
    """Input for the command if required."""

    name: str
    """Command name (e.g., `create_plan`, `research_codebase`)."""
