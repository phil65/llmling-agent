"""AggregatingServer for managing multiple servers with unified interface."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

from llmling_agent.log import get_logger
from llmling_agent_server.base import BaseServer


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from starlette.routing import Route

    from llmling_agent import AgentPool

# Type-safe server status literals
ServerStatus = Literal["not_initialized", "initialized", "running", "failed", "stopped"]


@dataclass(frozen=True)
class ServerInfo:
    """Type-safe server information."""

    name: str
    server_type: type[BaseServer]
    status: ServerStatus
    is_http: bool = False


logger = get_logger(__name__)


class AggregatingServer(BaseServer):
    """Server that aggregates multiple servers with unified interface.

    Manages multiple server instances (MCP, OpenAI API, ACP, AG-UI, A2A, etc.) as a single
    coordinated unit. All servers share the same AgentPool and are started/stopped
    together while maintaining the same BaseServer interface.

    Supports two modes of operation:

    1. **Separate mode** (default): Each server runs independently on its own port
    2. **Unified HTTP mode**: All HTTP servers (AG-UI, A2A, OpenAI API) are combined
       into a single Starlette application on one port

    The unified mode is more efficient as it:
    - Uses a single uvicorn instance
    - Shares connection resources
    - Simplifies deployment (one port to expose)

    Example (separate mode):
        ```python
        servers = [
            AGUIServer(pool, port=8002),
            A2AServer(pool, port=8001),
            MCPServer(pool, mcp_config),
        ]
        aggregating_server = AggregatingServer(pool, servers)
        ```

    Example (unified HTTP mode):
        ```python
        servers = [
            AGUIServer(pool),  # port ignored in unified mode
            A2AServer(pool),   # port ignored in unified mode
        ]
        aggregating_server = AggregatingServer(
            pool, servers,
            unified_http=True,
            unified_host="localhost",
            unified_port=8000,
        )
        # All routes accessible at http://localhost:8000
        # AG-UI: /agui/{agent_name}
        # A2A: /a2a/{agent_name}
        ```
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        servers: Sequence[BaseServer],
        *,
        name: str | None = None,
        raise_exceptions: bool = False,
        unified_http: bool = False,
        unified_host: str = "localhost",
        unified_port: int = 8000,
    ) -> None:
        """Initialize aggregating server.

        Args:
            pool: AgentPool to be managed by this server
            servers: Sequence of servers to aggregate
            name: Server name for logging (auto-generated if None)
            raise_exceptions: Whether to raise exceptions during server start
            unified_http: Whether to combine HTTP servers into single app
            unified_host: Host for unified HTTP server (only used if unified_http=True)
            unified_port: Port for unified HTTP server (only used if unified_http=True)
        """
        if not servers:
            msg = "At least one server must be provided"
            raise ValueError(msg)

        super().__init__(pool, name=name, raise_exceptions=raise_exceptions)

        self.servers = list(servers)
        self.exit_stack = AsyncExitStack()
        self._initialized_servers: list[BaseServer] = []

        # Unified HTTP mode settings
        self.unified_http = unified_http
        self.unified_host = unified_host
        self.unified_port = unified_port
        self._unified_app: Any = None

    async def __aenter__(self) -> Self:
        """Initialize aggregating server and all child servers."""
        await super().__aenter__()

        self.log.info("Initializing aggregated servers", count=len(self.servers))

        try:
            for server in self.servers:
                try:
                    initialized_server = await self.exit_stack.enter_async_context(server)
                    self._initialized_servers.append(initialized_server)
                    self.log.info("Initialized server", server_name=server.name)
                except Exception:
                    self.log.exception("Failed to initialize server", server_name=server.name)
                    if self.raise_exceptions:
                        raise

            if not self._initialized_servers:
                msg = "No servers were successfully initialized"
                raise RuntimeError(msg)  # noqa: TRY301

            self.log.info(
                "All servers initialized",
                successful=len(self._initialized_servers),
                failed=len(self.servers) - len(self._initialized_servers),
            )

        except Exception:
            await self.exit_stack.aclose()
            raise

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup all servers and base server resources."""
        self.log.info("Shutting down aggregated servers")

        await self.exit_stack.aclose()
        self._initialized_servers.clear()

        await super().__aexit__(exc_type, exc_val, exc_tb)

        self.log.info("Aggregated servers shutdown complete")

    def _is_http_server(self, server: BaseServer) -> bool:
        """Check if server is an HTTP-based server."""
        from llmling_agent_server.http_server import HTTPServer

        return isinstance(server, HTTPServer)

    def _get_http_servers(self) -> list[BaseServer]:
        """Get all HTTP-based servers."""
        return [s for s in self._initialized_servers if self._is_http_server(s)]

    def _get_non_http_servers(self) -> list[BaseServer]:
        """Get all non-HTTP servers."""
        return [s for s in self._initialized_servers if not self._is_http_server(s)]

    async def _collect_all_routes(self) -> list[Route]:
        """Collect routes from all HTTP servers with appropriate prefixes.

        Returns:
            Combined list of routes from all HTTP servers
        """
        from starlette.routing import Route

        from llmling_agent_server.http_server import HTTPServer

        all_routes: list[Route] = []

        for server in self._get_http_servers():
            if not isinstance(server, HTTPServer):
                continue

            # Get routes with server's own prefix
            routes = await server.get_routes()
            prefix = server.get_route_prefix()

            # If server doesn't have a prefix, generate one from class name
            if not prefix:
                # AGUIServer -> /agui, A2AServer -> /a2a, etc.
                class_name = server.__class__.__name__
                prefix = "/" + class_name.replace("Server", "").lower()

            # Apply prefix to routes
            for route in routes:
                if isinstance(route, Route):
                    new_path = f"{prefix}{route.path}"
                    all_routes.append(
                        Route(
                            new_path,
                            route.endpoint,
                            methods=route.methods,
                            name=f"{server.name}_{route.name}" if route.name else None,
                        )
                    )
                else:
                    all_routes.append(route)

            self.log.debug(
                "Collected routes from server",
                server=server.name,
                prefix=prefix,
                route_count=len(routes),
            )

        return all_routes

    async def _create_unified_app(self) -> Any:
        """Create unified Starlette application with all HTTP routes.

        Returns:
            Starlette application with combined routes
        """
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        all_routes = await self._collect_all_routes()

        # Add root endpoint that lists all servers and routes
        async def list_all_routes(request: Any) -> Any:
            """List all available routes from all servers."""
            server_info = []
            for server in self._get_http_servers():
                routes = await server.get_routes() if hasattr(server, "get_routes") else []
                prefix = server.get_route_prefix() if hasattr(server, "get_route_prefix") else ""
                if not prefix:
                    prefix = "/" + server.__class__.__name__.replace("Server", "").lower()

                server_info.append({
                    "name": server.name,
                    "type": server.__class__.__name__,
                    "prefix": prefix,
                    "routes": [f"{prefix}{r.path}" for r in routes if isinstance(r, Route)],
                })

            return JSONResponse({
                "servers": server_info,
                "total_routes": len(all_routes),
                "mode": "unified",
            })

        all_routes.append(Route("/", list_all_routes, methods=["GET"]))

        app = Starlette(debug=False, routes=all_routes)
        self.log.info(
            "Created unified HTTP app",
            total_routes=len(all_routes),
            http_servers=len(self._get_http_servers()),
        )
        return app

    async def _start_unified_http(self) -> None:
        """Start unified HTTP server with combined routes."""
        import uvicorn

        self._unified_app = await self._create_unified_app()

        config = uvicorn.Config(
            app=self._unified_app,
            host=self.unified_host,
            port=self.unified_port,
            log_level="info",
        )

        server = uvicorn.Server(config)

        self.log.info(
            "Starting unified HTTP server",
            host=self.unified_host,
            port=self.unified_port,
        )

        await server.serve()

    async def _start_async(self) -> None:
        """Start all initialized servers concurrently."""
        if not self._initialized_servers:
            self.log.warning("No initialized servers to start")
            return

        self.log.info(
            "Starting aggregated servers",
            count=len(self._initialized_servers),
            unified_http=self.unified_http,
        )

        if self.unified_http:
            # In unified mode: combine HTTP servers, start non-HTTP separately
            non_http_servers = self._get_non_http_servers()
            http_servers = self._get_http_servers()

            server_tasks: list[tuple[BaseServer | None, asyncio.Task[None] | None]] = []

            # Start non-HTTP servers in background
            for server in non_http_servers:
                try:
                    server.start_background()
                    server_tasks.append((server, server._server_task))
                    self.log.info("Started non-HTTP server in background", server=server.name)
                except Exception:
                    self.log.exception("Failed to start non-HTTP server", server=server.name)
                    if self.raise_exceptions:
                        raise

            # Create unified HTTP task if we have HTTP servers
            unified_task: asyncio.Task[None] | None = None
            if http_servers:
                unified_task = self.task_manager.create_task(
                    self._start_unified_http(), name="unified-http"
                )
                server_tasks.append((None, unified_task))
                self.log.info(
                    "Started unified HTTP server",
                    http_server_count=len(http_servers),
                )

            # Wait for any task to complete
            tasks = [task for _, task in server_tasks if task is not None]
            if tasks:
                await self._wait_for_tasks(tasks, server_tasks)

        else:
            # Separate mode: start each server independently
            await self._start_separate_servers()

    async def _start_separate_servers(self) -> None:
        """Start all servers in separate mode (each on its own port)."""
        server_tasks: list[tuple[BaseServer, asyncio.Task[None] | None]] = []

        for server in self._initialized_servers:
            try:
                server.start_background()
                server_tasks.append((server, server._server_task))
                self.log.info("Started server in background", server=server.name)
            except Exception:
                self.log.exception("Failed to start server", server=server.name)
                if self.raise_exceptions:
                    for started_server, _ in server_tasks:
                        try:
                            started_server.stop()
                        except Exception:
                            self.log.exception(
                                "Error stopping server", server_name=started_server.name
                            )
                    raise

        if not server_tasks:
            msg = "No servers were successfully started"
            raise RuntimeError(msg)

        self.log.info(
            "All servers started",
            successful=len(server_tasks),
            failed=len(self._initialized_servers) - len(server_tasks),
        )

        tasks = [task for _, task in server_tasks if task is not None]
        if tasks:
            await self._wait_for_tasks(tasks, server_tasks)

    async def _wait_for_tasks(
        self,
        tasks: list[asyncio.Task[None]],
        server_tasks: list[tuple[BaseServer | None, asyncio.Task[None] | None]],
    ) -> None:
        """Wait for server tasks and handle shutdown."""
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if task.exception():
                    self.log.error("Server task failed", error=task.exception())
                else:
                    self.log.info("Server task completed")

            # Stop all other servers gracefully
            for server, _ in server_tasks:
                if server is not None:
                    try:
                        server.stop()
                    except Exception:
                        self.log.exception("Error stopping server")

            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        except asyncio.CancelledError:
            for server, _ in server_tasks:
                if server is not None:
                    try:
                        server.stop()
                    except Exception:
                        self.log.exception("Error stopping server during cancellation")
            raise

    def add_server(self, server: BaseServer) -> None:
        """Add a server to the aggregation.

        Args:
            server: Server to add

        Raises:
            RuntimeError: If aggregating server is currently running
        """
        if self.is_running:
            msg = "Cannot add server while aggregating server is running"
            raise RuntimeError(msg)

        self.servers.append(server)
        self.log.info("Added server to aggregation", server=server.name)

    def remove_server(self, server: BaseServer) -> None:
        """Remove a server from the aggregation.

        Args:
            server: Server to remove

        Raises:
            RuntimeError: If aggregating server is currently running
            ValueError: If server is not in aggregation
        """
        if self.is_running:
            msg = "Cannot remove server while aggregating server is running"
            raise RuntimeError(msg)

        try:
            self.servers.remove(server)
            self.log.info("Removed server from aggregation", server=server.name)
        except ValueError as e:
            msg = f"Server {server.name} not found in aggregation"
            raise ValueError(msg) from e

    def get_server(self, name: str) -> BaseServer | None:
        """Get a server by name from the aggregation.

        Args:
            name: Server name to find

        Returns:
            Server instance or None if not found
        """
        all_servers = self.servers + self._initialized_servers
        for server in all_servers:
            if server.name == name:
                return server
        return None

    def list_servers(self) -> list[ServerInfo]:
        """List all servers in the aggregation with their status.

        Returns:
            List of type-safe ServerInfo objects
        """
        return [
            ServerInfo(
                name=server.name,
                server_type=type(server),
                status=self._get_server_status(server),
                is_http=self._is_http_server(server),
            )
            for server in self.servers
        ]

    def get_server_status(self) -> dict[str, ServerStatus]:
        """Get status of all servers.

        Returns:
            Dict mapping server names to their type-safe status
        """
        return {server.name: self._get_server_status(server) for server in self.servers}

    def _get_server_status(self, server: BaseServer) -> ServerStatus:
        """Get type-safe status for a specific server."""
        if server in self._initialized_servers:
            return "running" if server.is_running else "initialized"
        return "not_initialized"

    @property
    def initialized_server_count(self) -> int:
        """Number of successfully initialized servers."""
        return len(self._initialized_servers)

    @property
    def running_server_count(self) -> int:
        """Number of currently running servers."""
        return sum(1 for server in self._initialized_servers if server.is_running)

    @property
    def unified_base_url(self) -> str:
        """Get base URL for unified HTTP server (only valid in unified mode)."""
        return f"http://{self.unified_host}:{self.unified_port}"

    def __repr__(self) -> str:
        """String representation of aggregating server."""
        mode = "unified" if self.unified_http else "separate"
        return (
            f"AggregatingServer(name={self.name}, "
            f"mode={mode}, "
            f"servers={len(self.servers)}, "
            f"initialized={len(self._initialized_servers)}, "
            f"running={self.running_server_count})"
        )
