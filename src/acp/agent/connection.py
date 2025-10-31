"""Agent ACP Connection."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Self

from acp.agent.protocol import Agent
from acp.client.protocol import Client
from acp.connection import Connection
from acp.exceptions import RequestError
from acp.schema import (
    AuthenticateRequest,
    CancelNotification,
    CreateTerminalRequest,
    CreateTerminalResponse,
    CustomRequest,
    CustomResponse,
    InitializeRequest,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
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
        AgentRequest,
        ClientResponse,
        CreateTerminalRequest,
        KillTerminalCommandRequest,
        ReadTextFileRequest,
        ReleaseTerminalRequest,
        RequestPermissionRequest,
        SessionNotification,
        TerminalOutputRequest,
        WaitForTerminalExitRequest,
        WriteTextFileRequest,
    )
    from acp.schema.agent_responses import AgentResponse


class AgentSideConnection(Client):
    """Agent-side connection.

    Use when you implement the Agent and need to talk to a Client.

    Args:
        to_agent: factory that receives this connection and returns your Agent
        input: asyncio.StreamWriter (local -> peer)
        output: asyncio.StreamReader (peer -> local)
    """

    async def handle_request(self, request: AgentRequest) -> ClientResponse:
        """Not used.

        AgentSideConnection sends requests to clients, doesn't handle them.
        """
        msg = "AgentSideConnection doesn't handle requests - it sends them"
        raise NotImplementedError(msg)

    async def handle_notification(self, notification: SessionNotification[Any]) -> None:
        """Not used.

        AgentSideConnection sends notifications to clients, doesn't handle them.
        """
        msg = "AgentSideConnection doesn't handle notifications - it sends them"
        raise NotImplementedError(msg)

    def __init__(
        self,
        to_agent: Callable[[AgentSideConnection], Agent],
        input_stream: asyncio.StreamWriter,
        output_stream: asyncio.StreamReader,
        observers: list[StreamObserver] | None = None,
        *,
        debug_file: str | None = None,
    ) -> None:
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
    async def session_update(self, params: SessionNotification[Any]) -> None:
        dct = params.model_dump(by_alias=True, exclude_none=True)
        await self._conn.send_notification("session/update", dct)

    async def request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        method = "session/request_permission"
        resp = await self._conn.send_request(method, dct)
        return RequestPermissionResponse.model_validate(resp)

    async def read_text_file(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("fs/read_text_file", dct)
        return ReadTextFileResponse.model_validate(resp)

    async def write_text_file(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request("fs/write_text_file", dct)
        return WriteTextFileResponse.model_validate(r)

    # async def createTerminal(self, params: CreateTerminalRequest) -> TerminalHandle:
    async def create_terminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/create", dct)
        #  resp = CreateTerminalResponse.model_validate(resp)
        #  return TerminalHandle(resp.terminal_id, params.session_id, self._conn)
        return CreateTerminalResponse.model_validate(resp)

    async def custom_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a custom request method."""
        return await self._conn.send_request(f"_{method}", params)

    async def custom_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a custom notification method."""
        await self._conn.send_notification(f"_{method}", params)

    async def terminal_output(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/output", dct)
        return TerminalOutputResponse.model_validate(resp)

    async def release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/release", dct)
        return ReleaseTerminalResponse.model_validate(resp)

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/wait_for_exit", dct)
        return WaitForTerminalExitResponse.model_validate(resp)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request("terminal/kill", dct)
        return KillTerminalCommandResponse.model_validate(resp)

    async def close(self) -> None:
        await self._conn.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


async def _agent_handler(  # noqa: PLR0911
    agent: Agent,
    method: AgentMethod | str,
    params: dict[str, Any] | None,
    is_notification: bool,
) -> AgentResponse | dict[str, Any] | None:
    match method:
        case "initialize":
            init_request = InitializeRequest.model_validate(params)
            return await agent.handle_request(init_request)
        case "session/new":
            new_request = NewSessionRequest.model_validate(params)
            return await agent.handle_request(new_request)
        case "session/load":
            load_request = LoadSessionRequest.model_validate(params)
            return await agent.handle_request(load_request)
        case "session/set_mode":
            mode_request = SetSessionModeRequest.model_validate(params)
            return await agent.handle_request(mode_request)
        case "session/prompt":
            prompt_request = PromptRequest.model_validate(params)
            return await agent.handle_request(prompt_request)
        case "session/cancel":
            cancel_notification = CancelNotification.model_validate(params)
            await agent.cancel(cancel_notification)
            return None
        case "session/set_model":
            model_request = SetSessionModelRequest.model_validate(params)
            return await agent.handle_request(model_request)
        case "authenticate":
            auth_request = AuthenticateRequest.model_validate(params)
            return await agent.handle_request(auth_request)
        case str() if method.startswith("_") and is_notification:
            # Custom notifications - fire and forget
            custom_request = CustomRequest(method=method[1:], data=params or {})
            await agent.handle_request(custom_request)
            return None
        case str() if method.startswith("_"):
            # Custom requests - expect response
            custom_request = CustomRequest(method=method[1:], data=params or {})
            response = await agent.handle_request(custom_request)
            if isinstance(response, CustomResponse):
                return response.data
            return None
        case _:
            raise RequestError.method_not_found(method)
