"""Agent ACP Connection."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Self

import logfire

from acp.agent.protocol import Agent
from acp.client.protocol import Client
from acp.connection import Connection
from acp.exceptions import RequestError
from acp.schema import (
    AuthenticateRequest,
    CancelNotification,
    CreateTerminalRequest,
    CreateTerminalResponse,
    InitializeRequest,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionNotification,
    SetSessionModelRequest,
    SetSessionModeRequest,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from acp.task import DebuggingMessageStateStore


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from acp.agent.protocol import Agent
    from acp.connection import StreamObserver
    from acp.schema import (
        AgentMethod,
        CreateTerminalRequest,
        InitializeResponse,
        KillTerminalCommandRequest,
        ListSessionsResponse,
        LoadSessionResponse,
        NewSessionResponse,
        PromptResponse,
        ReadTextFileRequest,
        ReleaseTerminalRequest,
        RequestPermissionRequest,
        SessionNotification,
        TerminalOutputRequest,
        WaitForTerminalExitRequest,
        WriteTextFileRequest,
    )


class AgentSideConnection(Client):
    """Agent-side connection.

    Use when you implement the Agent and need to talk to a Client.

    Args:
        to_agent: factory that receives this connection and returns your Agent
        input: asyncio.StreamWriter (local -> peer)
        output: asyncio.StreamReader (peer -> local)
    """

    def __init__(
        self,
        to_agent: Callable[[AgentSideConnection], Agent],
        input_stream: asyncio.StreamWriter,
        output_stream: asyncio.StreamReader,
        observers: list[StreamObserver] | None = None,
        *,
        debug_file: str | None = None,
    ) -> None:
        """Initialize the agent-side connection.

        Args:
            to_agent: factory that receives this connection and returns your Agent
            input_stream: asyncio.StreamWriter (local -> peer)
            output_stream: asyncio.StreamReader (peer -> local)
            observers: list of StreamObserver instances to observe the connection
            debug_file: path to a file to write debug information to
        """
        agent = to_agent(self)
        handler = partial(_agent_handler, agent)
        store = DebuggingMessageStateStore(debug_file=debug_file) if debug_file else None
        self._conn = Connection(
            handler,
            input_stream,
            output_stream,
            state_store=store,
            observers=observers,
        )

    # client-bound methods (agent -> client)
    async def session_update(self, params: SessionNotification) -> None:
        """Send session update notification."""
        dct = params.model_dump(by_alias=True, exclude_none=True)
        await self._conn.send_notification("session/update", dct)

    async def request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Request permission from the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        method = "session/request_permission"
        resp = await self._conn.send_request(method, dct)
        return RequestPermissionResponse.model_validate(resp)

    async def read_text_file(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Read text file from the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("fs/read_text_file", dct)
        return ReadTextFileResponse.model_validate(resp)

    async def write_text_file(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        """Write text file to the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request("fs/write_text_file", dct)
        return WriteTextFileResponse.model_validate(r)

    # async def createTerminal(self, params: CreateTerminalRequest) -> TerminalHandle:
    async def create_terminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:
        """Create a terminal on the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/create", dct)
        #  resp = CreateTerminalResponse.model_validate(resp)
        #  return TerminalHandle(resp.terminal_id, params.session_id, self._conn)
        return CreateTerminalResponse.model_validate(resp)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an extension method on the client."""
        return await self._conn.send_request(f"_{method}", params)  # type: ignore[no-any-return]

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send an extension notification to the client."""
        await self._conn.send_notification(f"_{method}", params)

    async def terminal_output(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Show terminal output on the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/output", dct)
        return TerminalOutputResponse.model_validate(resp)

    async def release_terminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:
        """Release a terminal on the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/release", dct)
        return ReleaseTerminalResponse.model_validate(resp)

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit on the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/wait_for_exit", dct)
        return WaitForTerminalExitResponse.model_validate(resp)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        """Kill a terminal on the client."""
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/kill", dct)
        return KillTerminalCommandResponse.model_validate(resp)

    async def close(self) -> None:
        """Close the connection."""
        await self._conn.close()

    async def __aenter__(self) -> Self:
        """Enter the async context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context (close the connection)."""
        await self.close()


@logfire.instrument(r"Handle Agent Request {method}")
async def _agent_handler(  # noqa: PLR0911
    agent: Agent,
    method: AgentMethod | str,
    params: dict[str, Any] | None,
    is_notification: bool,
) -> (
    NewSessionResponse
    | InitializeResponse
    | PromptResponse
    | LoadSessionResponse
    | ListSessionsResponse
    | dict[str, Any]
    | None
):
    """Handle an agent request."""
    match method:
        case "initialize":
            initialize_request = InitializeRequest.model_validate(params)
            return await agent.initialize(initialize_request)
        case "session/new":
            new_session_request = NewSessionRequest.model_validate(params)
            return await agent.new_session(new_session_request)
        case "session/load":
            load_request = LoadSessionRequest.model_validate(params)
            return await agent.load_session(load_request)
        case "session/list":
            list_request = ListSessionsRequest.model_validate(params)
            return await agent.list_sessions(list_request)
        case "session/set_mode":
            set_mode_request = SetSessionModeRequest.model_validate(params)
            return (
                session_resp.model_dump(by_alias=True, exclude_none=True)
                if (session_resp := await agent.set_session_mode(set_mode_request))
                else {}
            )
        case "session/prompt":
            prompt_request = PromptRequest.model_validate(params)
            return await agent.prompt(prompt_request)
        case "session/cancel":
            cancel_notification = CancelNotification.model_validate(params)
            await agent.cancel(cancel_notification)
            return None
        case "session/set_model":
            set_model_request = SetSessionModelRequest.model_validate(params)
            return (
                model_result.model_dump(by_alias=True, exclude_none=True)
                if (model_result := await agent.set_session_model(set_model_request))
                else {}
            )
        case "authenticate":
            p = AuthenticateRequest.model_validate(params)
            result = await agent.authenticate(p)
            return result.model_dump(by_alias=True, exclude_none=True) if result else {}
        case str() if method.startswith("_") and is_notification:
            await agent.ext_notification(method[1:], params or {})
            return None
        case str() if method.startswith("_"):
            return await agent.ext_method(method[1:], params or {})
        case _:
            raise RequestError.method_not_found(method)
