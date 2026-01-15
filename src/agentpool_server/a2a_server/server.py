"""A2A server implementation for agentpool pool.

This module provides a server that exposes all agents in an AgentPool
via the A2A (Agent-to-Agent) protocol, with each agent accessible at its own route.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from agentpool.log import get_logger
from agentpool_server.http_server import HTTPServer


if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    from agentpool import AgentPool


logger = get_logger(__name__)

DEFAULT_PORT = 8001


class A2AServer(HTTPServer):
    """A2A server for exposing pool agents on separate routes.

    Provides a unified HTTP server that exposes all agents in the pool via
    the A2A protocol. Each agent is accessible at `/{agent_name}` route.

    Example:
        ```python
        pool = AgentPool(manifest)

        server = A2AServer(pool, host="localhost", port=8001)

        async with server:
            async with server.run_context():
                # Agents accessible at:
                # POST http://localhost:8001/agent1
                # POST http://localhost:8001/agent2
                await do_other_work()
        ```
    """

    def __init__(
        self,
        pool: AgentPool,
        *,
        name: str | None = None,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        raise_exceptions: bool = False,
    ) -> None:
        """Initialize A2A server.

        Args:
            pool: AgentPool containing available agents
            name: Optional server name (auto-generated if None)
            host: Host to bind server to
            port: Port to bind server to
            raise_exceptions: Whether to raise exceptions during server start
        """
        super().__init__(pool, name=name, host=host, port=port, raise_exceptions=raise_exceptions)

    async def get_routes(self) -> list[Route]:
        """Get Starlette routes for A2A protocol.

        Returns:
            List of Route objects for each agent plus root listing endpoint
        """
        from fasta2a import FastA2A  # type: ignore[import-untyped]
        from fasta2a.broker import InMemoryBroker  # type: ignore[import-untyped]
        from fasta2a.storage import InMemoryStorage  # type: ignore[import-untyped]
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        from agentpool_server.a2a_server.agent_worker import AgentWorker, worker_lifespan

        routes: list[Route] = []
        # Create route for each agent in the pool
        for agent_name in self.pool.all_agents:

            async def agent_handler(request: Request, agent_name: str = agent_name) -> Response:
                """Handle A2A requests for a specific agent."""
                try:
                    # Get the agent from pool
                    agent = self.pool.all_agents.get(agent_name)
                    if agent is None:
                        error = {"error": f"Agent '{agent_name}' not found"}
                        return JSONResponse(error, status_code=404)
                    # Get the underlying pydantic-ai agentlet and convert to A2A app
                    storage = InMemoryStorage()
                    broker = InMemoryBroker()
                    worker = AgentWorker(agent=agent, broker=broker, storage=storage)
                    lifespan = partial(worker_lifespan, worker=worker, agent=agent)
                    a2a_app = FastA2A(
                        storage=storage,
                        broker=broker,
                        name=agent.name,
                        lifespan=lifespan,
                    )
                    # ASGI apps don't return a value, they write to send()
                    await a2a_app(request.scope, request.receive, request._send)
                    return Response()

                except Exception as e:
                    self.log.exception("Error handling A2A request", agent=agent_name)
                    return JSONResponse({"error": str(e)}, status_code=500)

            # A2A protocol routes
            routes.append(Route(f"/{agent_name}", agent_handler, methods=["POST"]))
            routes.append(
                Route(
                    f"/{agent_name}/.well-known/agent-card.json",
                    agent_handler,
                    methods=["HEAD", "GET", "OPTIONS"],
                )
            )
            routes.append(Route(f"/{agent_name}/docs", agent_handler, methods=["GET"]))
            self.log.debug("Registered A2A routes", agent=agent_name, route=f"/{agent_name}")

        # Add root endpoint that lists available agents
        async def list_agents(request: Request) -> Response:
            """List all available agents."""
            from starlette.responses import JSONResponse

            agent_list = [
                {
                    "name": name,
                    "route": f"/{name}",
                    "agent_card": f"/{name}/.well-known/agent-card.json",
                    "docs": f"/{name}/docs",
                    "model": agent.model_name,
                }
                for name, agent in self.pool.all_agents.items()
            ]
            return JSONResponse({
                "agents": agent_list,
                "count": len(agent_list),
                "protocol": "A2A",
                "version": "0.3.0",
            })

        routes.append(Route("/", list_agents, methods=["GET"]))
        self.log.info("Created A2A routes", agent_count=len(self.pool.all_agents))
        return routes

    def get_agent_url(self, agent_name: str) -> str:
        """Get the URL for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Full URL for the agent's A2A endpoint
        """
        return f"{self.base_url}/{agent_name}"

    def get_agent_card_url(self, agent_name: str) -> str:
        """Get the agent card URL for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Full URL for the agent's A2A card endpoint
        """
        return f"{self.base_url}/{agent_name}/.well-known/agent-card.json"

    def list_agent_routes(self) -> dict[str, dict[str, str]]:
        """List all agent routes.

        Returns:
            Dictionary mapping agent names to their URLs
        """
        return {
            name: {
                "endpoint": self.get_agent_url(name),
                "agent_card": self.get_agent_card_url(name),
                "docs": f"{self.base_url}/{name}/docs",
            }
            for name in self.pool.all_agents
        }
