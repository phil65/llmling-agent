from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Literal

from pydantic import Field

from acp.schema.base import Response
from acp.schema.capabilities import AgentCapabilities
from acp.schema.common import AuthMethod, Implementation  # noqa: TC001
from acp.schema.session_state import SessionModelState, SessionModeState  # noqa: TC001


StopReason = Literal[
    "end_turn",
    "max_tokens",
    "max_turn_requests",
    "refusal",
    "cancelled",
]


class SetSessionModelResponse(Response):
    """**UNSTABLE**.

    This capability is not part of the spec yet.

    Response to `session/set_model` method.
    """


class NewSessionResponse(Response):
    """Response from creating a new session.

    See protocol docs: [Creating a Session](https://agentclientprotocol.com/protocol/session-setup#creating-a-session)
    """

    models: SessionModelState | None = None
    """**UNSTABLE**

    This capability is not part of the spec yet.

    Initial model state if supported by the Agent
    """

    modes: SessionModeState | None = None
    """Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """

    session_id: str
    """Unique identifier for the created session.

    Used in all subsequent requests for this conversation.
    """


class LoadSessionResponse(Response):
    """Response from loading an existing session."""

    models: SessionModelState | None = None
    """**UNSTABLE**

    This capability is not part of the spec yet.

    Initial model state if supported by the Agent
    """

    modes: SessionModeState | None = None
    """Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """


class SetSessionModeResponse(Response):
    """Response to `session/set_mode` method."""


class PromptResponse(Response):
    """Response from processing a user prompt.

    See protocol docs: [Check for Completion](https://agentclientprotocol.com/protocol/prompt-turn#4-check-for-completion)
    """

    stop_reason: StopReason
    """Indicates why the agent stopped processing the turn."""


class AuthenticateResponse(Response):
    """Response to authenticate method."""


class InitializeResponse(Response):
    """Response from the initialize method.

    Contains the negotiated protocol version and agent capabilities.

    See protocol docs: [Initialization](https://agentclientprotocol.com/protocol/initialization)
    """

    agent_capabilities: AgentCapabilities | None = Field(
        default_factory=AgentCapabilities
    )
    """Capabilities supported by the agent."""

    agent_info: Implementation | None = None
    """Information about the Agent name and version sent to the Client.


    Note: in future versions of the protocol, this will be required."""

    auth_methods: Sequence[AuthMethod] | None = Field(default_factory=list)
    """Authentication methods supported by the agent."""

    protocol_version: int = Field(ge=0, le=65535)
    """The protocol version the client specified if supported by the agent.

    Or the latest protocol version supported by the agent.
    The client should disconnect, if it doesn't support this version.
    """


AgentResponse = (
    InitializeResponse
    | AuthenticateResponse
    | NewSessionResponse
    | LoadSessionResponse
    | SetSessionModeResponse
    | PromptResponse
    | SetSessionModelResponse
)
