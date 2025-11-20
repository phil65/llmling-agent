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

    from llmling_agent import AgentPool

# Type-safe server status literals
ServerStatus = Literal["not_initialized", "initialized", "running", "failed", "stopped"]


@dataclass(frozen=True)
class ServerInfo:
    """Type-safe server information."""

    name: str
    server_type: type[BaseServer]
    status: ServerStatus


logger = get_logger(__name__)


class AggregatingServer(BaseServer):
    """Server that aggregates multiple servers with unified interface.

    Manages multiple server instances (MCP, OpenAI API, ACP, etc.) as a single
    coordinated unit. All servers share the same AgentPool and are started/stopped
    together while maintaining the same BaseServer interface.

    The aggregating server properly manages the lifecycle of all child servers:
    - Initializes all servers during __aenter__ using AsyncExitStack
    - Starts all servers concurrently during _start_async
    - Automatically cleans up all servers during __aexit

    Example:
        ```python
        pool = AgentPool(manifest)

        servers = [
            MCPServer(pool, mcp_config),
            OpenAIServer(pool, host="localhost", port=8000),
            ResponsesServer(pool, host="localhost", port=8001),
        ]

        aggregating_server = AggregatingServer(pool, servers)

        # Use like any other server
        async with aggregating_server:
            async with aggregating_server.run_context():
                # All servers running in background
                await do_other_work()
        ```
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        servers: Sequence[BaseServer],
        *,
        name: str | None = None,
        raise_exceptions: bool = False,
    ) -> None:
        """Initialize aggregating server.

        Args:
            pool: AgentPool to be managed by this server
            servers: Sequence of servers to aggregate
            name: Server name for logging (auto-generated if None)
            raise_exceptions: Whether to raise exceptions during server start
        """
        if not servers:
            msg = "At least one server must be provided"
            raise ValueError(msg)

        # Initialize base with the shared pool
        super().__init__(pool, name=name, raise_exceptions=raise_exceptions)

        self.servers = list(servers)
        self.exit_stack = AsyncExitStack()
        self._initialized_servers: list[BaseServer] = []

    async def __aenter__(self) -> Self:
        """Initialize aggregating server and all child servers."""
        # Initialize base server (pool)
        await super().__aenter__()

        self.log.info("Initializing aggregated servers", count=len(self.servers))

        try:
            # Initialize all child servers using exit stack
            for server in self.servers:
                try:
                    initialized_server = await self.exit_stack.enter_async_context(server)
                    self._initialized_servers.append(initialized_server)
                    self.log.info("Initialized server")
                except Exception:
                    self.log.exception("Failed to initialize server")
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
            # If initialization fails, cleanup what we've done so far
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

        # Exit stack will automatically cleanup all initialized servers
        await self.exit_stack.aclose()
        self._initialized_servers.clear()

        # Cleanup base server
        await super().__aexit__(exc_type, exc_val, exc_tb)

        self.log.info("Aggregated servers shutdown complete")

    async def _start_async(self) -> None:
        """Start all initialized servers concurrently."""
        if not self._initialized_servers:
            self.log.warning("No initialized servers to start")
            return

        self.log.info("Starting aggregated servers", count=len(self._initialized_servers))

        # Start all servers in background
        server_tasks = []
        for server in self._initialized_servers:
            try:
                server.start_background()
                server_tasks.append((server, server._server_task))
                self.log.info("Started server in background")
            except Exception:
                self.log.exception("Failed to start server")
                if self.raise_exceptions:
                    # Stop any servers we've already started
                    for started_server, _task in server_tasks:
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

        try:
            # Wait for any server to complete (indicates shutdown or failure)
            tasks = [task for _, task in server_tasks if task is not None]
            if tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Log which server completed first
                for task in done:
                    if task.exception():
                        self.log.error("Server failed", error=task.exception())
                    else:
                        self.log.info("Server completed")

                # Stop all other servers gracefully
                for server, _ in server_tasks:
                    try:
                        server.stop()
                    except Exception:
                        self.log.exception("Error stopping server")

                # Wait for all tasks to complete
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        except asyncio.CancelledError:
            # If we're cancelled, stop all servers
            for server, _ in server_tasks:
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
        self.log.info("Added server to aggregation")

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
            self.log.info("Removed server from aggregation")
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
        # Check both configured and initialized servers
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

    def __repr__(self) -> str:
        """String representation of aggregating server."""
        return (
            f"AggregatingServer(name={self.name}, "
            f"servers={len(self.servers)}, "
            f"initialized={len(self._initialized_servers)}, "
            f"running={self.running_server_count})"
        )
