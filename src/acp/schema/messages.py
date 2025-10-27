"""Message schema definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from acp.base import Schema
from acp.schema.agent_requests import AgentRequest
from acp.schema.agent_responses import AgentResponse
from acp.schema.client_requests import ClientRequest
from acp.schema.client_responses import ClientResponse
from acp.schema.notifications import CancelNotification, SessionNotification


class JsonRPCMessage(Schema):
    """JSON-RPC 2.0 message."""

    jsonrpc: Literal["2.0"] = Field(default="2.0", init=False)
    """JSON RPC Messsage."""


class ClientNotificationMessage(JsonRPCMessage):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    method: str
    """Method name."""

    params: CancelNotification | Any | None = None
    """Agent notification parameters."""


class ClientResponseMessage(JsonRPCMessage):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    id: int | str | None = None
    """JSON RPC Request Id."""

    result: ClientResponse | Any
    """All possible responses that a client can send to an agent.

    This enum is used internally for routing RPC responses. You typically won't need
    to use this directly - the responses are handled automatically by the connection.
    These are responses to the corresponding `AgentRequest` variants."""


class AgentResponseMessage(JsonRPCMessage):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    id: int | str | None = None
    """JSON RPC Request Id."""

    result: AgentResponse | Any
    """All possible responses that an agent can send to a client.

    This enum is used internally for routing RPC responses. You typically won't need
    to use this directly - the responses are handled automatically by the connection.
    These are responses to the corresponding `ClientRequest` variants."""


class ClientRequestMessage(JsonRPCMessage):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    id: int | str | None = None
    """JSON RPC Request Id."""

    method: str
    """Method name."""

    params: ClientRequest | Any | None = None
    """Client request parameters."""


class AgentRequestMessage(JsonRPCMessage):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    id: int | str | None = None
    """JSON RPC Request Id."""

    method: str
    """Method name."""

    params: AgentRequest | Any | None = None
    """Agent request parameters."""


class AgentNotificationMessage(JsonRPCMessage):
    """A message (request, response, or notification) with `"jsonrpc": "2.0"`.

    Specified as [required by JSON-RPC 2.0 Specification][1].

    [1]: https://www.jsonrpc.org/specification#compatibility
    """

    method: str
    """Method name."""

    params: SessionNotification | Any | None = None
    """Agent notification parameters."""
