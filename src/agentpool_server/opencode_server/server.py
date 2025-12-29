"""OpenCode-compatible FastAPI server.

This server implements the OpenCode API endpoints to allow OpenCode SDK clients
to interact with AgentPool agents.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request  # noqa: TC002
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from agentpool import AgentPool
from agentpool_server.opencode_server.routes import (
    agent_router,
    app_router,
    config_router,
    file_router,
    global_router,
    message_router,
    pty_router,
    session_router,
)
from agentpool_server.opencode_server.state import ServerState


class OpenCodeJSONResponse(JSONResponse):
    """Custom JSON response that excludes None values (like OpenCode does)."""

    def render(self, content: Any) -> bytes:
        return super().render(jsonable_encoder(content, exclude_none=True))


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


VERSION = "0.1.0"


def create_app(
    *,
    pool: AgentPool[Any],
    agent_name: str | None = None,
    working_dir: str | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        pool: AgentPool for session persistence and agent access.
        agent_name: Name of the agent to use for handling messages.
                   If None, uses the first agent in the pool.
        working_dir: Working directory for file operations. Defaults to cwd.

    Returns:
        Configured FastAPI application.

    Raises:
        ValueError: If specified agent_name not found or pool has no agents.
    """
    # Resolve the agent from the pool
    if agent_name:
        agent = pool.all_agents.get(agent_name)
        if agent is None:
            msg = f"Agent '{agent_name}' not found in pool"
            raise ValueError(msg)
    else:
        # Use first agent as default
        agent = next(iter(pool.all_agents.values()), None)
        if agent is None:
            msg = "Pool has no agents"
            raise ValueError(msg)

    state = ServerState(
        working_dir=working_dir or str(Path.cwd()),
        pool=pool,
        agent=agent,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        yield
        # Shutdown

    app = FastAPI(
        title="OpenCode-Compatible API",
        description="AgentPool server with OpenCode API compatibility",
        version=VERSION,
        lifespan=lifespan,
        default_response_class=OpenCodeJSONResponse,
    )

    # Add CORS middleware (required for OpenCode TUI)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state on app for access in routes
    app.state.server_state = state

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = await request.body()
        print(f"Validation error for {request.url}")
        print(f"Body: {body.decode()}")
        print(f"Errors: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "body": body.decode()},
        )

    # Register routers
    app.include_router(global_router)
    app.include_router(app_router)
    app.include_router(config_router)
    app.include_router(session_router)
    app.include_router(message_router)
    app.include_router(file_router)
    app.include_router(agent_router)
    app.include_router(pty_router)

    # OpenAPI doc redirect
    @app.get("/doc")
    async def get_doc() -> RedirectResponse:
        """Redirect to OpenAPI docs."""
        return RedirectResponse(url="/docs")

    return app


class OpenCodeServer:
    """OpenCode-compatible server wrapper.

    Provides a convenient interface for running the server.
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        *,
        host: str = "127.0.0.1",
        port: int = 4096,
        agent_name: str | None = None,
        working_dir: str | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            pool: AgentPool for session persistence and agent access.
            host: Host to bind to.
            port: Port to listen on.
            agent_name: Name of the agent to use for handling messages.
            working_dir: Working directory for file operations.
        """
        self.host = host
        self.port = port
        self.pool = pool
        self.agent_name = agent_name
        self.working_dir = working_dir
        self._app: FastAPI | None = None

    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = create_app(
                pool=self.pool,
                agent_name=self.agent_name,
                working_dir=self.working_dir,
            )
        return self._app

    def run(self) -> None:
        """Run the server (blocking)."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    async def run_async(self) -> None:
        """Run the server asynchronously."""
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="websockets-sansio")
        server = uvicorn.Server(config)
        await server.serve()


def run_server(
    pool: AgentPool[Any],
    *,
    host: str = "127.0.0.1",
    port: int = 4096,
    agent_name: str | None = None,
    working_dir: str | None = None,
) -> None:
    """Run the OpenCode-compatible server.

    Args:
        pool: AgentPool for session persistence and agent access.
        host: Host to bind to.
        port: Port to listen on.
        agent_name: Name of the agent to use for handling messages.
        working_dir: Working directory for file operations.
    """
    server = OpenCodeServer(
        pool,
        host=host,
        port=port,
        agent_name=agent_name,
        working_dir=working_dir,
    )
    server.run()


if __name__ == "__main__":
    from agentpool import config_resources

    pool = AgentPool(config_resources.CLAUDE_CODE_ASSISTANT)
    run_server(pool)
