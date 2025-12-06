"""Simple pydantic-ai AG-UI server for testing AGUIAgent."""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from starlette.applications import Starlette
from starlette.routing import Route


if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response


# Create a simple pydantic-ai agent with TestModel
from pydantic_ai.models.test import TestModel


# Configure TestModel with custom responses
test_model = TestModel(
    custom_output_text="The answer is 4.",
)
test_agent = Agent(
    test_model,
    system_prompt="You are a helpful test assistant. Keep responses brief and clear.",
)


async def handle_agent_request(request: Request) -> Response:
    """Handle AG-UI protocol requests."""
    from pydantic_ai.ag_ui import handle_ag_ui_request

    return await handle_ag_ui_request(test_agent, request)


# Create Starlette app
app = Starlette(
    routes=[
        Route("/agent/run", handle_agent_request, methods=["POST"]),
    ],
)


async def main(port: int = 8765) -> None:
    """Run the AG-UI test server."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(main(args.port))
