"""Client request schema definitions."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Any

from pydantic import Field

from acp.schema.base import Request
from acp.schema.capabilities import ClientCapabilities
from acp.schema.common import Implementation  # noqa: TC001
from acp.schema.content_blocks import ContentBlock  # noqa: TC001
from acp.schema.mcp import McpServer  # noqa: TC001


class CustomRequest(Request):
    """Request for custom/extension methods."""

    method: str
    """The custom method name (without underscore prefix)."""

    data: dict[str, Any]
    """The method parameters."""


class NewSessionRequest(Request):
    """Request parameters for creating a new session.

    See protocol docs: [Creating a Session](https://agentclientprotocol.com/protocol/session-setup#creating-a-session)
    """

    cwd: str
    """The working directory for this session. Must be an absolute path."""

    mcp_servers: Sequence[McpServer]
    """List of MCP (Model Context Protocol) servers the agent should connect to."""


class LoadSessionRequest(Request):
    """Request parameters for loading an existing session.

    Only available if the Agent supports the `loadSession` capability.

    See protocol docs: [Loading Sessions](https://agentclientprotocol.com/protocol/session-setup#loading-sessions)
    """

    cwd: str
    """The working directory for this session."""

    mcp_servers: Sequence[McpServer]
    """List of MCP servers to connect to for this session."""

    session_id: str
    """The ID of the session to load."""


class SetSessionModeRequest(Request):
    """Request parameters for setting a session mode."""

    mode_id: str
    """The ID of the mode to set."""

    session_id: str
    """The ID of the session to set the mode for."""


class PromptRequest(Request):
    """Request parameters for sending a user prompt to the agent.

    Contains the user's message and any additional context.

    See protocol docs: [User Message](https://agentclientprotocol.com/protocol/prompt-turn#1-user-message)
    """

    prompt: Sequence[ContentBlock]
    """The blocks of content that compose the user's message.

    As a baseline, the Agent MUST support [`ContentBlock::Text`] and
    [`ContentBlock::ResourceContentBlock`],
    while other variants are optionally enabled via [`PromptCapabilities`].

    The Client MUST adapt its interface according to [`PromptCapabilities`].

    The client MAY include referenced pieces of context as either
    [`ContentBlock::Resource`] or [`ContentBlock::ResourceContentBlock`].

    When available, [`ContentBlock::Resource`] is preferred
    as it avoids extra round-trips and allows the message to include
    pieces of context from sources the agent may not have access to.
    """

    session_id: str
    """The ID of the session to send this user message to."""


class SetSessionModelRequest(Request):
    """**UNSTABLE**.

    This capability is not part of the spec yet.

    Request parameters for setting a session model.
    """

    model_id: str
    """The ID of the model to set."""

    session_id: str
    """The ID of the session to set the model for."""


class InitializeRequest(Request):
    """Request parameters for the initialize method.

    Sent by the client to establish connection and negotiate capabilities.

    See protocol docs: [Initialization](https://agentclientprotocol.com/protocol/initialization)
    """

    client_capabilities: ClientCapabilities | None = Field(
        default_factory=ClientCapabilities
    )
    """Capabilities supported by the client."""

    client_info: Implementation | None = None
    """Information about the Client name and version sent to the Agent.

    Note: in future versions of the protocol, this will be required.
    """

    protocol_version: int
    """The latest protocol version supported by the client."""


class AuthenticateRequest(Request):
    """Request parameters for the authenticate method.

    Specifies which authentication method to use.
    """

    method_id: str
    """The ID of the authentication method to use.

    Must be one of the methods advertised in the initialize response.
    """


ClientRequest = (
    InitializeRequest
    | AuthenticateRequest
    | NewSessionRequest
    | LoadSessionRequest
    | SetSessionModeRequest
    | PromptRequest
    | SetSessionModelRequest
    | CustomRequest
)
