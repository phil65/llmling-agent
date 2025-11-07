"""AggregatingServer for managing multiple servers with unified interface."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent_server.base import BaseServer


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent import AgentPool


logger = get_logger(__name__)


class AggregatingServer(BaseServer):
    """Server that aggregates multiple servers with unified interface.

    Manages multiple server instances (MCP, OpenAI API, ACP, etc.) as a single
    coordinated unit. All servers share the same AgentPool and are started/stopped
    together while maintaining the same BaseServer interface.

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
        self._server_tasks: list[asyncio.Task[None]] = []
        self._failed_servers: set[str] = set()

        # Configure servers to not manage pool lifecycle since we do it
        for server in self.servers:
            if hasattr(server, "manage_pool"):
                server.manage_pool = False
            # Bind their task managers to ours for centralized cleanup
            if hasattr(server, "task_manager"):
                # Each server keeps its own task manager for isolation
                pass

    @property
    def is_running(self) -> bool:
        """Check if any server is currently running."""
        # If we have server tasks and any are not done, we're running
        if self._server_tasks:
            return any(not task.done() for task in self._server_tasks)
        # Fall back to base class check
        return super().is_running

    async def _start_async(self) -> None:
        """Start all aggregated servers concurrently."""
        if not self.servers:
            return

        self.log.info("Starting aggregated servers", count=len(self.servers))

        # Initialize all servers first (this doesn't start them)
        initialized_servers = []
        for server in self.servers:
            try:
                await server.__aenter__()
                initialized_servers.append(server)
                self.log.info("Initialized server", server_name=server.name)
            except Exception:
                self.log.exception("Failed to initialize server", server_name=server.name)
                self._failed_servers.add(server.name)
                if self.raise_exceptions:
                    # Cleanup already initialized servers
                    for init_server in initialized_servers:
                        with contextlib.suppress(Exception):
                            await init_server.__aexit__(None, None, None)
                    raise

        # Start all initialized servers concurrently
        if initialized_servers:
            try:
                # Create tasks for each server's _start_async method
                self._server_tasks = [
                    self.task_manager.create_task(
                        server._start_async(), name=f"{server.name}-task"
                    )
                    for server in initialized_servers
                ]

                self.log.info(
                    "All servers started",
                    successful=len(initialized_servers),
                    failed=len(self._failed_servers),
                )

                # Wait for any server to complete (indicates shutdown or failure)
                done, pending = await asyncio.wait(
                    self._server_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Log which server completed first
                for task in done:
                    server_name = task.get_name()
                    if task.exception():
                        self.log.error(
                            "Server failed",
                            server_name=server_name,
                            error=task.exception(),
                        )
                    else:
                        self.log.info("Server completed", server_name=server_name)

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()

                # Wait for all tasks to complete
                await asyncio.gather(*pending, return_exceptions=True)

            finally:
                # Cleanup all servers
                for server in initialized_servers:
                    try:
                        await server.__aexit__(None, None, None)
                    except Exception:
                        self.log.exception(
                            "Error during server cleanup", server_name=server.name
                        )

    async def shutdown(self) -> None:
        """Shutdown all servers and cleanup resources."""
        self.log.info("Shutting down aggregated servers")

        # Cancel all server tasks
        for task in self._server_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._server_tasks:
            await asyncio.gather(*self._server_tasks, return_exceptions=True)

        self._server_tasks.clear()

        # Call parent cleanup
        await super().shutdown()

        self.log.info("Aggregated servers shutdown complete")

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

        # Configure server to not manage pool lifecycle
        if hasattr(server, "manage_pool"):
            server.manage_pool = False

        self.servers.append(server)
        self.log.info("Added server to aggregation", server_name=server.name)

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
            self.log.info("Removed server from aggregation", server_name=server.name)
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
        for server in self.servers:
            if server.name == name:
                return server
        return None

    def list_servers(self) -> list[tuple[str, type[BaseServer], bool]]:
        """List all servers in the aggregation.

        Returns:
            List of tuples: (name, server_type, is_failed)
        """
        return [
            (server.name, type(server), server.name in self._failed_servers)
            for server in self.servers
        ]

    def get_server_status(self) -> dict[str, str]:
        """Get status of all servers.

        Returns:
            Dict mapping server names to their status
        """
        status = {}
        for server in self.servers:
            if server.name in self._failed_servers:
                status[server.name] = "failed"
            elif self.is_running:
                # Check if server's task is still running
                server_task = next(
                    (
                        task
                        for task in self._server_tasks
                        if task.get_name().startswith(server.name)
                    ),
                    None,
                )
                if server_task and not server_task.done():
                    status[server.name] = "running"
                else:
                    status[server.name] = "stopped"
            else:
                status[server.name] = "stopped"
        return status

    def __repr__(self) -> str:
        """String representation of aggregating server."""
        return (
            f"AggregatingServer(name={self.name}, "
            f"servers={len(self.servers)}, "
            f"running={self.is_running})"
        )
