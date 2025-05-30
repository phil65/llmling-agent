"""OpenAI-compatible responses endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

from llmling_agent_server.responses_helpers import handle_request
from llmling_agent_server.responses_models import Response, ResponseRequest  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent import AgentPool


class ResponsesServer:
    """OpenAI-compatible /v1/responses endpoint."""

    def __init__(self, pool: AgentPool):
        self.pool = pool
        self.app = FastAPI()
        self.setup_routes()

    def verify_api_key(
        self, authorization: Annotated[str | None, Header(alias="Authorization")] = None
    ):
        """Verify API key if configured."""
        if not authorization:
            raise HTTPException(401, "Missing API key")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid authorization format")

    def setup_routes(self):
        """Set up API routes."""
        self.app.post("/v1/responses", dependencies=[Depends(self.verify_api_key)])(
            self.create_response
        )

    async def create_response(self, req_body: ResponseRequest) -> Response:
        """Handle response creation requests."""
        try:
            agent = self.pool.agents[req_body.model]
            return await handle_request(req_body, agent)
        except Exception as e:
            raise HTTPException(500, str(e)) from e


if __name__ == "__main__":
    import asyncio
    import logging

    import httpx
    import uvicorn

    from llmling_agent import AgentPool

    async def test_client():
        """Test the API with a direct HTTP request."""
        logger = logging.getLogger(__name__)

        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "http://localhost:8000/v1/responses",
                headers={"Authorization": "Bearer dummy"},
                json={
                    "model": "gpt-4o",
                    "input": "Tell me a three sentence bedtime story about a unicorn.",
                },
            )
            logger.info("Response: %s", response.text)

            if not response.is_success:
                logger.error("Error: %s", response.text)

    async def main():
        """Run server and test client."""
        pool = AgentPool[None]()
        await pool.add_agent("gpt-4o", model="openai:gpt-4")
        async with pool:
            server = ResponsesServer(pool)
            config = uvicorn.Config(
                server.app, host="0.0.0.0", port=8000, log_level="info"
            )
            server_instance = uvicorn.Server(config)
            server_task = asyncio.create_task(server_instance.serve())
            await asyncio.sleep(1)
            await test_client()

            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                logging.info("Server task cancelled.")

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    asyncio.run(main())
