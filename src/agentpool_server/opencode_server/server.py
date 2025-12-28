"""OpenCode-compatible FastAPI server.

This server implements the OpenCode API endpoints to allow OpenCode SDK clients
to interact with AgentPool agents.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from agentpool_server.opencode_server.routes import (
    agent_router,
    app_router,
    config_router,
    file_router,
    global_router,
    message_router,
    session_router,
)
from agentpool_server.opencode_server.state import ServerState


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

VERSION = "0.1.0"


def create_app(*, working_dir: str | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        working_dir: Working directory for file operations. Defaults to cwd.

    Returns:
        Configured FastAPI application.
    """
    state = ServerState(working_dir=working_dir or str(Path.cwd()))

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
    )

    # Store state on app for access in routes
    app.state.server_state = state

    # Register routers
    app.include_router(global_router)
    app.include_router(app_router)
    app.include_router(config_router)
    app.include_router(session_router)
    app.include_router(message_router)
    app.include_router(file_router)
    app.include_router(agent_router)

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
        *,
        host: str = "127.0.0.1",
        port: int = 4096,
        working_dir: str | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            working_dir: Working directory for file operations.
        """
        self.host = host
        self.port = port
        self.working_dir = working_dir
        self._app: FastAPI | None = None

    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = create_app(working_dir=self.working_dir)
        return self._app

    def run(self) -> None:
        """Run the server (blocking)."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    async def run_async(self) -> None:
        """Run the server asynchronously."""
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 4096,
    working_dir: str | None = None,
) -> None:
    """Run the OpenCode-compatible server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        working_dir: Working directory for file operations.
    """
    server = OpenCodeServer(host=host, port=port, working_dir=working_dir)
    server.run()


if __name__ == "__main__":
    run_server()
