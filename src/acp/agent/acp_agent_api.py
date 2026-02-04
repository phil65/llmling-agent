"""ACP Agent API for simplified client-to-agent interactions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp.schema import (
    AuthenticateRequest,
    CancelNotification,
    ForkSessionRequest,
    InitializeRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    ResumeSessionRequest,
    SetSessionConfigOptionRequest,
    SetSessionModelRequest,
    SetSessionModeRequest,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp.agent.protocol import Agent
    from acp.schema import (
        AuthenticateResponse,
        ContentBlock,
        ForkSessionResponse,
        InitializeResponse,
        ListSessionsResponse,
        LoadSessionResponse,
        NewSessionResponse,
        PromptResponse,
        ResumeSessionResponse,
        SetSessionConfigOptionResponse,
        SetSessionModelResponse,
        SetSessionModeResponse,
    )
    from acp.schema.mcp import McpServer


class ACPAgentAPI:
    """Thin wrapper for client-to-agent ACP interactions.

    Avoids manual instantiation of request/notification objects.
    """

    def __init__(self, connection: Agent) -> None:
        """Initialize agent API helper.

        Args:
            connection: The Agent protocol connection (e.g., ClientSideConnection)
        """
        self.connection = connection

    async def initialize(
        self,
        *,
        title: str,
        version: str,
        name: str,
        protocol_version: int = 1,
        terminal: bool = True,
        read_text_file: bool = True,
        write_text_file: bool = True,
    ) -> InitializeResponse:
        """Initialize the ACP connection."""
        request = InitializeRequest.create(
            title=title,
            version=version,
            name=name,
            protocol_version=protocol_version,
            terminal=terminal,
            read_text_file=read_text_file,
            write_text_file=write_text_file,
        )
        return await self.connection.initialize(request)

    async def new_session(
        self,
        cwd: str | None = None,
        mcp_servers: Sequence[McpServer] | None = None,
    ) -> NewSessionResponse:
        """Create a new ACP session."""
        request = NewSessionRequest(
            cwd=cwd or str(Path.cwd()),
            mcp_servers=list(mcp_servers) if mcp_servers else None,
        )
        return await self.connection.new_session(request)

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Sequence[McpServer] | None = None,
    ) -> LoadSessionResponse:
        """Load an existing session."""
        request = LoadSessionRequest(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=list(mcp_servers) if mcp_servers else None,
        )
        return await self.connection.load_session(request)

    async def list_sessions(
        self,
        cwd: str | None = None,
    ) -> ListSessionsResponse:
        """List available sessions."""
        request = ListSessionsRequest(cwd=cwd)
        return await self.connection.list_sessions(request)

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Sequence[McpServer] | None = None,
    ) -> ForkSessionResponse:
        """Fork an existing session."""
        request = ForkSessionRequest(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=list(mcp_servers) if mcp_servers else [],
        )
        return await self.connection.fork_session(request)

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Sequence[McpServer] | None = None,
    ) -> ResumeSessionResponse:
        """Resume a paused session."""
        request = ResumeSessionRequest(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=list(mcp_servers) if mcp_servers else [],
        )
        return await self.connection.resume_session(request)

    async def prompt(
        self,
        session_id: str,
        prompt: Sequence[ContentBlock],
    ) -> PromptResponse:
        """Send a prompt to the agent."""
        request = PromptRequest(session_id=session_id, prompt=list(prompt))
        return await self.connection.prompt(request)

    async def cancel(self, session_id: str) -> None:
        """Cancel the current operation in a session."""
        notification = CancelNotification(session_id=session_id)
        await self.connection.cancel(notification)

    async def set_session_mode(
        self,
        session_id: str,
        mode_id: str,
    ) -> SetSessionModeResponse | None:
        """Set the session mode."""
        request = SetSessionModeRequest(session_id=session_id, mode_id=mode_id)
        return await self.connection.set_session_mode(request)

    async def set_session_model(
        self,
        session_id: str,
        model_id: str,
    ) -> SetSessionModelResponse | None:
        """Set the session model."""
        request = SetSessionModelRequest(session_id=session_id, model_id=model_id)
        return await self.connection.set_session_model(request)

    async def set_session_config_option(
        self,
        session_id: str,
        config_id: str,
        value: str,
    ) -> SetSessionConfigOptionResponse | None:
        """Set a session configuration option."""
        request = SetSessionConfigOptionRequest(
            session_id=session_id,
            config_id=config_id,
            value=value,  # pyright: ignore[reportCallIssue]
        )
        return await self.connection.set_session_config_option(request)

    async def authenticate(
        self,
        method_id: str,
    ) -> AuthenticateResponse | None:
        """Authenticate with the agent."""
        request = AuthenticateRequest(method_id=method_id)
        return await self.connection.authenticate(request)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an extension method on the agent."""
        return await self.connection.ext_method(method, params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send an extension notification to the agent."""
        await self.connection.ext_notification(method, params)
