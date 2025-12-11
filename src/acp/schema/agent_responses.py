from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Any, Literal, Self

from pydantic import Field

from acp.schema.base import Response
from acp.schema.capabilities import AgentCapabilities
from acp.schema.common import AuthMethod, Implementation  # noqa: TC001
from acp.schema.session_state import (  # noqa: TC001
    SessionInfo,
    SessionModelState,
    SessionModeState,
)


class CustomResponse(Response):
    """Response for custom/extension methods."""

    data: dict[str, Any] | None = None
    """The method result data."""


StopReason = Literal[
    "end_turn",
    "max_tokens",
    "max_turn_requests",
    "refusal",
    "cancelled",
]


class SetSessionModelResponse(Response):
    """**UNSTABLE**: This capability is not part of the spec yet.

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


class ForkSessionResponse(Response):
    """**UNSTABLE**: This capability is not part of the spec yet.

    Response from forking an existing session.
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
    """Unique identifier for the newly created forked session."""


class ResumeSessionResponse(Response):
    """**UNSTABLE**: This capability is not part of the spec yet.

    Response from resuming an existing session.
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

    agent_capabilities: AgentCapabilities | None = Field(default_factory=AgentCapabilities)
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

    @classmethod
    def create(
        cls,
        name: str,
        title: str,
        version: str,
        protocol_version: int,
        load_session: bool | None = False,
        http_mcp_servers: bool = False,
        sse_mcp_servers: bool = False,
        audio_prompts: bool = False,
        embedded_context_prompts: bool = False,
        image_prompts: bool = False,
        list_sessions: bool = False,
        auth_methods: Sequence[AuthMethod] | None = None,
    ) -> Self:
        """Create an instance of AgentCapabilities.

        Args:
            name: The name of the agent.
            title: The title of the agent.
            version: The version of the agent.
            protocol_version: The protocol version of the agent.
            load_session: Whether the agent supports `session/load`.
            http_mcp_servers: Whether the agent supports HTTP MCP servers.
            sse_mcp_servers: Whether the agent supports SSE MCP servers.
            audio_prompts: Whether the agent supports audio prompts.
            embedded_context_prompts: Whether the agent supports embedded context prompts.
            image_prompts: Whether the agent supports image prompts.
            list_sessions: Whether the agent supports `session/list` (unstable).
            auth_methods: The authentication methods supported by the agent.
        """
        caps = AgentCapabilities.create(
            load_session=load_session,
            http_mcp_servers=http_mcp_servers,
            sse_mcp_servers=sse_mcp_servers,
            audio_prompts=audio_prompts,
            embedded_context_prompts=embedded_context_prompts,
            image_prompts=image_prompts,
            list_sessions=list_sessions,
        )
        return cls(
            agent_info=Implementation(name=name, title=title, version=version),
            protocol_version=protocol_version,
            agent_capabilities=caps,
            auth_methods=auth_methods,
        )


class ListSessionsResponse(Response):
    """**UNSTABLE**: This capability is not part of the spec yet.

    Response from listing sessions.
    """

    next_cursor: str | None = None
    """Opaque cursor token. If present, pass this in the next request's cursor parameter
    to fetch the next page. If absent, there are no more results."""

    sessions: Sequence[SessionInfo] = Field(default_factory=list)
    """Array of session information objects."""


AgentResponse = (
    InitializeResponse
    | AuthenticateResponse
    | NewSessionResponse
    | LoadSessionResponse
    | ForkSessionResponse
    | ResumeSessionResponse
    | ListSessionsResponse
    | SetSessionModeResponse
    | PromptResponse
    | SetSessionModelResponse
    | CustomResponse
)
