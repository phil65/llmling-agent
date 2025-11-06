"""OpenAI-compatible API server for LLMling agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import anyenv
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
import logfire

from llmling_agent.log import get_logger
from llmling_agent_server import BaseServer
from llmling_agent_server.completions.helpers import stream_response
from llmling_agent_server.completions.models import (
    ChatCompletionResponse,
    Choice,
    OpenAIMessage,
    OpenAIModelInfo,
)


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent_server.completions.models import ChatCompletionRequest

logger = get_logger(__name__)


class OpenAIServer(BaseServer):
    """OpenAI-compatible API server backed by LLMling agents."""

    def __init__(
        self,
        pool: AgentPool,
        *,
        name: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        cors: bool = True,
        docs: bool = True,
        raise_exceptions: bool = False,
    ):
        """Initialize OpenAI-compatible server.

        Args:
            pool: AgentPool containing available agents
            name: Optional Server name (auto-generated if None)
            host: Host to bind server to
            port: Port to bind server to
            cors: Whether to enable CORS middleware
            docs: Whether to enable API documentation endpoints
            raise_exceptions: Whether to raise exceptions during server start
        """
        super().__init__(pool, name=name, raise_exceptions=raise_exceptions)
        self.host = host
        self.port = port
        self.app = FastAPI()
        logfire.instrument_fastapi(self.app)

        if cors:
            from fastapi.middleware.cors import CORSMiddleware

            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        if not docs:
            self.app.docs_url = None
            self.app.redoc_url = None

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
        """Configure API routes."""
        self.app.get("/v1/models")(self.list_models)
        dep = Depends(self.verify_api_key)
        self.app.post("/v1/chat/completions", dependencies=[dep], response_model=None)(
            self.create_chat_completion
        )

    async def list_models(self) -> dict[str, Any]:
        """List available agents as models."""
        models = []
        for name, agent in self.pool.agents.items():
            info = OpenAIModelInfo(id=name, created=0, description=agent.description)
            models.append(info)
        return {"object": "list", "data": models}

    async def create_chat_completion(self, request: ChatCompletionRequest) -> Response:
        """Handle chat completion requests."""
        try:
            agent = self.pool.agents[request.model]
        except KeyError:
            raise HTTPException(404, f"Model {request.model} not found") from None

        # Just take the last message content - let agent handle history
        content = request.messages[-1].content or ""
        if request.stream:
            return StreamingResponse(
                stream_response(agent, content, request),
                media_type="text/event-stream",
            )
        try:
            response = await agent.run(content)
            message = OpenAIMessage(role="assistant", content=str(response.content))
            completion_response = ChatCompletionResponse(
                id=response.message_id,
                created=int(response.timestamp.timestamp()),
                model=request.model,
                choices=[Choice(message=message)],
                usage=response.cost_info.token_usage if response.cost_info else None,  # pyright: ignore
            )
            json = completion_response.model_dump_json()
            return Response(content=json, media_type="application/json")
        except Exception as e:
            self.log.exception("Error processing chat completion")
            raise HTTPException(500, f"Error: {e!s}") from e

    async def _start_async(self) -> None:
        """Start the server (blocking async - runs until stopped)."""
        import uvicorn

        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    import asyncio

    import httpx

    from llmling_agent import AgentPool

    async def test_client():
        """Test the API with a direct HTTP request."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                headers={"Authorization": "Bearer dummy"},
                json={
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": "Tell me a joke"}],
                    "stream": True,
                },
                timeout=30.0,  # Longer timeout for streaming
            )

            if response.is_success:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        chunk = anyenv.load_json(data, return_type=dict)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                print("\n")
            else:
                print("Response error:", response.text)

    async def main():
        """Run server and test client."""
        pool = AgentPool()
        await pool.add_agent("gpt-5-mini", model="openai:gpt-5-mini")
        async with (
            OpenAIServer(pool, host="0.0.0.0", port=8000) as server,
            server.run_context(),
        ):  # Server initializes pool and starts in background
            await asyncio.sleep(1)  # Wait for server to start
            await test_client()
        # Server automatically stopped

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    asyncio.run(main())
