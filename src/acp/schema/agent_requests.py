from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from pydantic import Field

from acp.base import Request
from acp.schema.common import EnvVariable  # noqa: TC001
from acp.schema.tool_call import PermissionOption, ToolCall  # noqa: TC001


class WriteTextFileRequest(Request):
    """Request to write content to a text file.

    Only available if the client supports the `fs.writeTextFile` capability.
    """

    content: str
    """The text content to write to the file."""

    path: str
    """Absolute path to the file to write."""

    session_id: str
    """The session ID for this request."""


class ReadTextFileRequest(Request):
    """Request to read content from a text file.

    Only available if the client supports the `fs.readTextFile` capability.
    """

    limit: int | None = Field(default=None, ge=0)
    """Maximum number of lines to read."""

    line: int | None = Field(default=None, ge=0)
    """Line number to start reading from (1-based)."""

    path: str
    """Absolute path to the file to read."""

    session_id: str
    """The session ID for this request."""


class TerminalOutputRequest(Request):
    """Request to get the current output and status of a terminal."""

    session_id: str
    """The session ID for this request."""

    terminal_id: str
    """The ID of the terminal to get output from."""


class WaitForTerminalExitRequest(Request):
    """Request to wait for a terminal command to exit."""

    session_id: str
    """The session ID for this request."""

    terminal_id: str
    """The ID of the terminal to wait for."""


class CreateTerminalRequest(Request):
    """Request to create a new terminal and execute a command."""

    args: Sequence[str] | None = None
    """Array of command arguments."""

    command: str
    """The command to execute."""

    cwd: str | None = None
    """Working directory for the command (absolute path)."""

    env: Sequence[EnvVariable] | None = None
    """Environment variables for the command."""

    output_byte_limit: int | None = Field(default=None, ge=0)
    """Maximum number of output bytes to retain.

    When the limit is exceeded, the Client truncates from the beginning of the output
    to stay within the limit.

    The Client MUST ensure truncation happens at a character boundary to maintain valid
    string output, even if this means the retained output is slightly less than the
    specified limit."""

    session_id: str
    """The session ID for this request."""


class KillTerminalCommandRequest(Request):
    """Request to kill a terminal command without releasing the terminal."""

    session_id: str
    """The session ID for this request."""

    terminal_id: str
    """The ID of the terminal to kill."""


class ReleaseTerminalRequest(Request):
    """Request to release a terminal and free its resources."""

    session_id: str
    """The session ID for this request."""

    terminal_id: str
    """The ID of the terminal to release."""


class RequestPermissionRequest(Request):
    """Request for user permission to execute a tool call.

    Sent when the agent needs authorization before performing a sensitive operation.

    See protocol docs: [Requesting Permission](https://agentclientprotocol.com/protocol/tool-calls#requesting-permission)
    """

    options: Sequence[PermissionOption]
    """Available permission options for the user to choose from."""

    session_id: str
    """The session ID for this request."""

    tool_call: ToolCall
    """Details about the tool call requiring permission."""


AgentRequest = (
    WriteTextFileRequest
    | ReadTextFileRequest
    | RequestPermissionRequest
    | CreateTerminalRequest
    | TerminalOutputRequest
    | ReleaseTerminalRequest
    | WaitForTerminalExitRequest
    | KillTerminalCommandRequest
)
