"""ACP Bridge - Proxy stdio ACP agents to streamable HTTP transport.

This module provides functionality to spawn a stdio-based ACP agent subprocess
and expose it via a streamable HTTP endpoint.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn

from acp.bridge.settings import BridgeSettings
from acp.client.connection import ClientSideConnection
from acp.client.implementations import NoOpClient
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
    SetSessionModelRequest,
    SetSessionModeRequest,
)
from acp.transports import spawn_stdio_transport


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Mapping
    from pathlib import Path

    from starlette.requests import Request

    from acp.schema import AgentMethod


logger = logging.getLogger(__name__)


class ACPBridge:
    """Bridge that proxies stdio ACP agents to streamable HTTP."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        settings: BridgeSettings | None = None,
    ) -> None:
        """Initialize the ACP bridge.

        Args:
            command: Command to spawn the ACP agent.
            args: Arguments for the command.
            env: Environment variables for the subprocess.
            cwd: Working directory for the subprocess.
            settings: Bridge server settings.
        """
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.settings = settings or BridgeSettings()
        self._connection: ClientSideConnection | None = None
        self._process: asyncio.subprocess.Process | None = None

    async def _handle_acp_request(self, request: Request) -> Response:
        """Handle incoming ACP JSON-RPC requests."""
        if self._connection is None:
            response = {"jsonrpc": "2.0", "error": {"code": -32603, "message": "Not connected"}}
            return JSONResponse(response, status_code=503)

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            response = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}
            return JSONResponse(response, status_code=400)

        method = body.get("method")
        params = body.get("params")
        request_id = body.get("id")
        is_notification = request_id is None

        if method is None:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
                status_code=400,
            )

        try:
            result = await self._dispatch_to_agent(method, params, is_notification)
            if is_notification:
                return Response(status_code=204)
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as exc:
            logger.exception("Error dispatching request to agent")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(exc)},
                },
                status_code=500,
            )

    async def _dispatch_to_agent(  # noqa: PLR0911
        self,
        method: AgentMethod,
        params: dict[str, Any] | None,
        is_notification: bool,
    ) -> Any:
        """Dispatch a request to the connected agent."""
        if self._connection is None:
            msg = "No agent connection"
            raise RuntimeError(msg)
        match method:
            case "initialize":
                init_request = InitializeRequest.model_validate(params)
                init_resp = await self._connection.initialize(init_request)
                return init_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/new":
                new_session_request = NewSessionRequest.model_validate(params)
                new_session_resp = await self._connection.new_session(new_session_request)
                return new_session_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/load":
                load_session_request = LoadSessionRequest.model_validate(params)
                load_sessoin_resp = await self._connection.load_session(load_session_request)
                return load_sessoin_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/list":
                list_sessions_request = ListSessionsRequest.model_validate(params)
                list_session_resp = await self._connection.list_sessions(list_sessions_request)
                return list_session_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/fork":
                fork_session_request = ForkSessionRequest.model_validate(params)
                fork_session_resp = await self._connection.fork_session(fork_session_request)
                return fork_session_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/resume":
                resume_session_request = ResumeSessionRequest.model_validate(params)
                resume_session_resp = await self._connection.resume_session(resume_session_request)
                return resume_session_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/prompt":
                prompt_request = PromptRequest.model_validate(params)
                prompt_resp = await self._connection.prompt(prompt_request)
                return prompt_resp.model_dump(by_alias=True, exclude_none=True)
            case "session/cancel":
                cancel_request = CancelNotification.model_validate(params)
                await self._connection.cancel(cancel_request)
                return None
            case "session/set_mode":
                set_mode_request = SetSessionModeRequest.model_validate(params)
                resp = await self._connection.set_session_mode(set_mode_request)
                return resp.model_dump(by_alias=True, exclude_none=True) if resp else {}
            case "session/set_model":
                set_model_request = SetSessionModelRequest.model_validate(params)
                set_model_resp = await self._connection.set_session_model(set_model_request)
                return (
                    set_model_resp.model_dump(by_alias=True, exclude_none=True)
                    if set_model_resp
                    else {}
                )
            case "authenticate":
                authenticate_request = AuthenticateRequest.model_validate(params)
                authenticate_resp = await self._connection.authenticate(authenticate_request)
                return (
                    authenticate_resp.model_dump(by_alias=True, exclude_none=True)
                    if authenticate_resp
                    else {}
                )
            case str() if method.startswith("_") and is_notification:
                await self._connection.ext_notification(method[1:], params or {})
                return None
            case str() if method.startswith("_"):
                return await self._connection.ext_method(method[1:], params or {})
            case _:
                msg = f"Method not found: {method}"
                raise ValueError(msg)

    async def _handle_status(self, request: Request) -> Response:
        """Health check endpoint."""
        status = "connected" if self._connection else "disconnected"
        return JSONResponse({"status": status, "command": self.command, "args": self.args})

    def _create_app(self) -> Starlette:
        """Create the Starlette application."""
        routes = [
            Route("/acp", endpoint=self._handle_acp_request, methods=["POST"]),
            Route("/status", endpoint=self._handle_status, methods=["GET"]),
        ]

        middleware: list[Middleware] = []
        if self.settings.allow_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self.settings.allow_origins,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            )

        return Starlette(
            debug=(self.settings.log_level == "DEBUG"),
            routes=routes,
            middleware=middleware,
        )

    async def run(self) -> None:
        """Run the bridge server."""
        async with contextlib.AsyncExitStack() as stack:
            # Spawn the stdio agent subprocess
            logger.info("Spawning ACP agent: %s %s", self.command, " ".join(self.args))
            reader, writer, process = await stack.enter_async_context(
                spawn_stdio_transport(self.command, *self.args, env=self.env, cwd=self.cwd)
            )
            self._process = process

            # Create client connection to the agent
            def client_factory(agent: Any) -> NoOpClient:
                return NoOpClient()

            self._connection = ClientSideConnection(client_factory, writer, reader)
            stack.push_async_callback(self._connection.close)
            # Create and run the HTTP server
            app = self._create_app()
            config = uvicorn.Config(
                app,
                host=self.settings.host,
                port=self.settings.port,
                log_level=self.settings.log_level.lower(),
            )
            server = uvicorn.Server(config)
            url = f"http://{self.settings.host}:{self.settings.port}"
            logger.info("ACP Bridge serving at %s/acp", url)
            await server.serve()
