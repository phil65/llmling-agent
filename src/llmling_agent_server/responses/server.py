"""OpenAI-compatible responses endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
import logfire

from llmling_agent_server.responses.helpers import handle_request
from llmling_agent_server.responses.models import Response, ResponseRequest  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent import AgentPool


class ResponsesServer:
    """OpenAI-compatible /v1/responses endpoint."""

    def __init__(self, pool: AgentPool):
        self.pool = pool
        self.app = FastAPI()
        logfire.instrument_fastapi(self.app)
        self.setup_routes()

    def verify_api_key(
        self,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ):
        """Verify API key if configured."""
        if not authorization:
            raise HTTPException(401, "Missing API key")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid authorization format")

    def setup_routes(self):
        """Set up API routes."""
        deps = Depends(self.verify_api_key)
        self.app.post("/v1/responses", dependencies=[deps])(self.create_response)

    async def create_response(self, req_body: ResponseRequest) -> Response:
        """Handle response creation requests."""
        try:
            agent = self.pool.agents[req_body.model]
            return await handle_request(req_body, agent)
        except Exception as e:
            raise HTTPException(500, str(e)) from e

    async def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the server."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server_instance = uvicorn.Server(config)
        await server_instance.serve()


if __name__ == "__main__":
    import asyncio

    import httpx
    import uvicorn

    from llmling_agent import AgentPool

    async def test_client():
        """Test the API with a direct HTTP request."""
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "http://localhost:8000/v1/responses",
                headers={"Authorization": "Bearer dummy"},
                json={
                    "model": "gpt-5-nano",
                    "input": "Tell me a three sentence bedtime story about a unicorn.",
                },
            )
            print("Response:", response.text)

            if not response.is_success:
                print("Error:", response.text)

    async def main():
        """Run server and test client."""
        pool = AgentPool[None]()
        await pool.add_agent("gpt-5-nano", model="openai:gpt-5-nano")
        async with pool:
            server = ResponsesServer(pool)
            server_task = asyncio.create_task(server.serve())
            await asyncio.sleep(1)
            await test_client()

            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                print("Server task cancelled.")

    asyncio.run(main())
