"""Common schema definitions."""

from __future__ import annotations

from acp.base import AnnotatedObject, Schema


class EnvVariable(AnnotatedObject):
    """An environment variable to set when launching an MCP server."""

    name: str
    """The name of the environment variable."""

    value: str
    """The value to set for the environment variable."""


class Implementation(Schema):
    """Describes the name and version of an MCP implementation.

    Includes an optional title for UI representation.
    """

    name: str
    """Intended for programmatic or logical use.

    Can be used as a display name fallback if title isn't present."""

    title: str | None = None
    """Intended for UI and end-user contexts.

    Optimized to be human-readable and easily understood.
    If not provided, the name should be used for display."""

    version: str
    """Version of the implementation.

    Can be displayed to the user or used for debugging or metrics purposes."""


class AuthMethod(AnnotatedObject):
    """Describes an available authentication method."""

    description: str | None = None
    """Optional description providing more details about this authentication method."""

    id: str
    """Unique identifier for this authentication method."""

    name: str
    """Human-readable name of the authentication method."""
