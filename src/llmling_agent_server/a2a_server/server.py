"""A2A server implementation following BaseServer pattern."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
import uuid

from fasta2a.schema import (
    AgentCapabilities,
    AgentCard,
    a2a_request_ta,
    a2a_response_ta,
    agent_card_ta,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import FileResponse, Response
from starlette.routing import Route
import uvicorn

from llmling_agent.log import get_logger
from llmling_agent_server import BaseServer


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.routing import Route
    from starlette.types import ExceptionHandler, Receive, Scope, Send

    from llmling_agent import Agent, AgentPool


logger = get_logger(__name__)

# AgentWorker output type needs to be invariant for use
# in both parameter and return positions
WorkerOutputT = TypeVar("WorkerOutputT")

AgentDepsT = Any
OutputDataT = Any


class FastA2A(Starlette):
    """The main class for the FastA2A library."""

    def __init__(
        self,
        *,
        storage,
        broker,
        # Agent card
        name: str | None = None,
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        description: str | None = None,
        provider=None,
        skills=None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan=None,
    ):
        if lifespan is None:
            lifespan = _default_lifespan

        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

        self.name = name or "My Agent"
        self.url = url
        self.version = version
        self.description = description
        self.provider = provider
        self.skills = skills or []
        # NOTE: For now, I don't think there's any reason to support any other
        # input/output modes.
        self.default_input_modes = ["application/json"]
        self.default_output_modes = ["application/json"]

        self.task_manager = TaskManager(broker=broker, storage=storage)

        # Setup
        self._agent_card_json_schema: bytes | None = None
        self.router.add_route(
            "/.well-known/agent-card.json",
            self._agent_card_endpoint,
            methods=["HEAD", "GET", "OPTIONS"],
        )
        self.router.add_route("/", self._agent_run_endpoint, methods=["POST"])
        self.router.add_route("/docs", self._docs_endpoint, methods=["GET"])

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not self.task_manager.is_running:
            msg = "TaskManager was not properly initialized."
            raise RuntimeError(msg)
        await super().__call__(scope, receive, send)

    async def _agent_card_endpoint(self, request: Request) -> Response:
        if self._agent_card_json_schema is None:
            agent_card = AgentCard(
                name=self.name,
                description=self.description or "An AI agent exposed as an A2A agent.",
                url=self.url,
                version=self.version,
                protocol_version="0.3.0",
                skills=self.skills,
                default_input_modes=self.default_input_modes,
                default_output_modes=self.default_output_modes,
                capabilities=AgentCapabilities(
                    streaming=False,
                    push_notifications=False,
                    state_transition_history=False,
                ),
            )
            if self.provider is not None:
                agent_card["provider"] = self.provider
            self._agent_card_json_schema = agent_card_ta.dump_json(
                agent_card, by_alias=True
            )
        return Response(
            content=self._agent_card_json_schema, media_type="application/json"
        )

    async def _docs_endpoint(self, request: Request) -> Response:
        """Serve the documentation interface."""
        docs_path = Path(__file__).parent / "static" / "docs.html"
        return FileResponse(docs_path, media_type="text/html")

    async def _agent_run_endpoint(self, request: Request) -> Response:
        """Main endpoint for the A2A server.

        Although the specification allows freedom of choice and implementation,
        I'm pretty sure about some decisions.

        1. The server will always either send a "submitted" or a "failed" on
           `message/send`. Never a "completed" on the first message.
        2. There are three possible ends for the task:
            2.1. The task was "completed" successfully.
            2.2. The task was "canceled".
            2.3. The task "failed".
        3. The server will send a "working" on the first chunk on
           `tasks/pushNotification/get`.
        """
        data = await request.body()
        a2a_request = a2a_request_ta.validate_json(data)

        if a2a_request["method"] == "message/send":
            jsonrpc_response = await self.task_manager.send_message(a2a_request)
        elif a2a_request["method"] == "tasks/get":
            jsonrpc_response = await self.task_manager.get_task(a2a_request)
        elif a2a_request["method"] == "tasks/cancel":
            jsonrpc_response = await self.task_manager.cancel_task(a2a_request)
        else:
            msg = f"Method {a2a_request['method']} not implemented."
            raise NotImplementedError(msg)
        return Response(
            content=a2a_response_ta.dump_json(jsonrpc_response, by_alias=True),
            media_type="application/json",
        )


class TaskManager:
    """Task manager for A2A server."""

    def __init__(self, *, broker, storage):
        self.broker = broker
        self.storage = storage
        self.is_running = False

    async def __aenter__(self):
        self.is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.is_running = False

    async def send_message(self, request):
        """Handle message/send requests."""
        task_id = str(uuid.uuid4())
        # Store task in storage for processing
        await self.storage.store_task(task_id, request)
        await self.broker.submit_task(task_id)
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"status": "submitted", "task_id": task_id},
        }

    async def get_task(self, request):
        """Handle tasks/get requests."""
        task_id = request.get("params", {}).get("task_id")
        task_status = await self.storage.get_task_status(task_id)
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": task_status,
        }

    async def cancel_task(self, request):
        """Handle tasks/cancel requests."""
        task_id = request.get("params", {}).get("task_id")
        await self.storage.cancel_task(task_id)
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"status": "cancelled"},
        }


class SimpleStorage:
    """Simple in-memory storage for A2A tasks."""

    def __init__(self):
        self.tasks = {}

    async def store_task(self, task_id: str, task_data):
        """Store a task."""
        self.tasks[task_id] = {
            "id": task_id,
            "status": "submitted",
            "data": task_data,
            "result": None,
            "error": None,
        }

    async def get_task(self, task_id: str):
        """Get task data."""
        task = self.tasks.get(task_id)
        return task["data"] if task else None

    async def get_task_status(self, task_id: str):
        """Get task status."""
        task = self.tasks.get(task_id)
        if not task:
            return {"status": "not_found"}
        return {
            "status": task["status"],
            "result": task["result"],
            "error": task["error"],
        }

    async def update_task_status(
        self, task_id: str, status: str, result=None, error=None
    ):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            if result is not None:
                self.tasks[task_id]["result"] = result
            if error is not None:
                self.tasks[task_id]["error"] = error

    async def cancel_task(self, task_id: str):
        """Cancel a task."""
        await self.update_task_status(task_id, "cancelled")


class SimpleBroker:
    """Simple in-memory broker for A2A tasks."""

    def __init__(self, worker=None):
        self.worker = worker
        self.tasks = []

    async def submit_task(self, task_id: str):
        """Submit a task for processing."""
        self.tasks.append(task_id)
        if self.worker:
            # Process immediately for simplicity
            await self.worker.run_task(task_id)

    def set_worker(self, worker):
        """Set the worker for this broker."""
        self.worker = worker


@asynccontextmanager
async def _default_lifespan(app: FastA2A) -> AsyncIterator[None]:
    async with app.task_manager:
        yield


@asynccontextmanager
async def worker_lifespan(
    app: FastA2A, worker, agent: Agent[AgentDepsT, OutputDataT]
) -> AsyncIterator[None]:
    """Custom lifespan that runs the worker during application startup."""
    async with app.task_manager, agent, worker.run():
        yield


@dataclass
class AgentWorker:
    """A worker that uses an agent to execute tasks."""

    agent: Agent[AgentDepsT, OutputDataT]
    storage: Any
    broker: Any = None

    async def run(self):
        """Run the worker."""
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def run_task(self, task_id: str) -> None:
        """Run a task using the agent."""
        task_data = await self.storage.get_task(task_id)
        if not task_data:
            msg = f"Task {task_id} not found"
            raise ValueError(msg)

        await self.storage.update_task_status(task_id, "working")

        try:
            # Extract message content from A2A request
            message_content = self._extract_message_content(task_data)

            # Run the agent
            result = await self.agent.run(message_content)

            # Convert result to A2A format
            a2a_result = self._convert_result_to_a2a(result)

            await self.storage.update_task_status(task_id, "completed", result=a2a_result)
        except Exception as e:  # noqa: BLE001
            await self.storage.update_task_status(task_id, "failed", error=str(e))

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        await self.storage.update_task_status(task_id, "cancelled")

    def _extract_message_content(self, task_data) -> str:
        """Extract message content from A2A task data."""
        # Simple implementation - extract text from first message part
        messages = task_data.get("params", {}).get("messages", [])
        if messages:
            parts = messages[0].get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    return part.get("text", "")
        return ""

    def _convert_result_to_a2a(self, result) -> dict:
        """Convert agent result to A2A format."""
        return {
            "messages": [
                {
                    "role": "agent",
                    "parts": [
                        {
                            "kind": "text",
                            "text": str(result.content),
                        }
                    ],
                }
            ]
        }


class A2AServer(BaseServer):
    """A2A server wrapper following BaseServer pattern.

    This server wraps the FastA2A Starlette application and adapts it
    to follow our standard BaseServer interface for consistent lifecycle management.
    """

    def __init__(
        self,
        pool: AgentPool,
        *,
        name: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        # A2A specific options
        agent_name: str | None = None,
        url: str | None = None,
        version: str = "1.0.0",
        description: str | None = None,
        # Server options
        raise_exceptions: bool = False,
        **kwargs: Any,
    ):
        """Initialize A2A server.

        Args:
            pool: AgentPool containing available agents
            name: Optional Server name (auto-generated if None)
            host: Host to bind server to
            port: Port to bind server to
            agent_name: Name for the A2A agent card
            url: URL for the A2A agent card
            version: Version for the A2A agent card
            description: Description for the A2A agent card
            raise_exceptions: Whether to raise exceptions during server start
            **kwargs: Additional arguments passed to FastA2A
        """
        super().__init__(pool, name=name, raise_exceptions=raise_exceptions)
        self.host = host
        self.port = port
        self.agent_name = agent_name or "LLMling Agent"
        self.url = url or f"http://{host}:{port}"
        self.version = version
        self.description = description
        self.kwargs = kwargs

        # Get the first agent from the pool for now
        # TODO: Support multiple agents or agent selection
        if not self.pool.agents:
            msg = "No agents available in pool"
            raise ValueError(msg)

        agent = next(iter(self.pool.agents.values()))

        # Create A2A components
        storage = SimpleStorage()
        worker = AgentWorker(agent=agent, storage=storage)
        broker = SimpleBroker(worker=worker)
        worker.broker = broker

        lifespan = partial(worker_lifespan, worker=worker, agent=agent)

        # Create the FastA2A app
        self._app = FastA2A(
            storage=storage,
            broker=broker,
            name=self.agent_name,
            url=self.url,
            version=self.version,
            description=self.description,
            lifespan=lifespan,
            **self.kwargs,
        )

    async def _start_async(self) -> None:
        """Start the A2A server (blocking async - runs until stopped)."""
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    @property
    def app(self):
        """Get the underlying FastA2A application."""
        return self._app


if __name__ == "__main__":
    import asyncio

    import httpx

    from llmling_agent import AgentPool

    async def test_client():
        """Test the A2A server with a basic request."""
        async with httpx.AsyncClient() as client:
            # Test agent card endpoint
            response = await client.get(
                "http://localhost:8000/.well-known/agent-card.json"
            )
            print("Agent card:", response.json())

            # Test docs endpoint
            response = await client.get("http://localhost:8000/docs")
            print("Docs status:", response.status_code)

    async def main():
        """Run server and test client."""
        pool = AgentPool()
        await pool.add_agent("test-agent", model="openai:gpt-4o-mini")

        async with (
            A2AServer(
                pool,
                host="0.0.0.0",
                port=8000,
                agent_name="Test LLMling Agent",
                description="A test agent for A2A protocol",
            ) as server,
            server.run_context(),
        ):
            await asyncio.sleep(1)  # Wait for server to start
            await test_client()

    asyncio.run(main())
