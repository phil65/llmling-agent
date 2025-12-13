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

from acp.client.connection import ClientSideConnection
from acp.client.implementations import NoOpClient
from acp.transports import spawn_stdio_transport


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Mapping
    from pathlib import Path

    from starlette.requests import Request

    from acp.bridge.settings import BridgeSettings


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
        from acp.bridge.settings import BridgeSettings

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
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32603, "message": "Not connected"}},
                status_code=503,
            )

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

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
        method: str,
        params: dict[str, Any] | None,
        is_notification: bool,
    ) -> Any:
        """Dispatch a request to the connected agent."""
        if self._connection is None:
            msg = "No agent connection"
            raise RuntimeError(msg)

        # Import schema types for request construction
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

        match method:
            case "initialize":
                req = InitializeRequest.model_validate(params)
                resp = await self._connection.initialize(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/new":
                req = NewSessionRequest.model_validate(params)
                resp = await self._connection.new_session(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/load":
                req = LoadSessionRequest.model_validate(params)
                resp = await self._connection.load_session(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/list":
                req = ListSessionsRequest.model_validate(params)
                resp = await self._connection.list_sessions(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/fork":
                req = ForkSessionRequest.model_validate(params)
                resp = await self._connection.fork_session(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/resume":
                req = ResumeSessionRequest.model_validate(params)
                resp = await self._connection.resume_session(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/prompt":
                req = PromptRequest.model_validate(params)
                resp = await self._connection.prompt(req)
                return resp.model_dump(by_alias=True, exclude_none=True)
            case "session/cancel":
                req = CancelNotification.model_validate(params)
                await self._connection.cancel(req)
                return None
            case "session/set_mode":
                req = SetSessionModeRequest.model_validate(params)
                resp = await self._connection.set_session_mode(req)
                return resp.model_dump(by_alias=True, exclude_none=True) if resp else {}
            case "session/set_model":
                req = SetSessionModelRequest.model_validate(params)
                resp = await self._connection.set_session_model(req)
                return resp.model_dump(by_alias=True, exclude_none=True) if resp else {}
            case "authenticate":
                req = AuthenticateRequest.model_validate(params)
                resp = await self._connection.authenticate(req)
                return resp.model_dump(by_alias=True, exclude_none=True) if resp else {}
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
        return JSONResponse({
            "status": "connected" if self._connection else "disconnected",
            "command": self.command,
            "args": self.args,
        })

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
                spawn_stdio_transport(
                    self.command,
                    *self.args,
                    env=self.env,
                    cwd=self.cwd,
                )
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
