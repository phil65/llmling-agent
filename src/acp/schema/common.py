"""Common schema definitions."""

from __future__ import annotations

from typing import Any

from acp.schema.base import AnnotatedObject, Schema


class EnvVariable(AnnotatedObject):
    """An environment variable to set when launching an MCP server."""

    name: str
    """The name of the environment variable."""

    value: str
    """The value to set for the environment variable."""


class Implementation(Schema):
    """Describes the name and version of an ACP implementation.

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


class Error(Schema):
    """JSON-RPC error object.

    Represents an error that occurred during method execution, following the
    JSON-RPC 2.0 error object specification with optional additional data.

    See protocol docs: [JSON-RPC Error Object](https://www.jsonrpc.org/specification#error_object)
    """

    code: int
    """A number indicating the error type that occurred.

    This must be an integer as defined in the JSON-RPC specification.
    """

    data: Any | None = None
    """Optional primitive or structured value that contains additional errorinformation.

    This may include debugging information or context-specific details.
    """

    message: str
    """A string providing a short description of the error.

    The message should be limited to a concise single sentence.
    """
