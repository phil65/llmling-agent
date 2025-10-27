"""Error schema definitions."""

from __future__ import annotations

from typing import Any, Literal

from acp.base import Schema


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


class ErrorMessage(Schema):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as
    [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    jsonrpc: Literal["2.0"] = "2.0"

    id: int | str | None = None
    """JSON RPC Request Id."""

    error: Error
