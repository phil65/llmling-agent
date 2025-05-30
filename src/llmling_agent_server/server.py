"""OpenAI-compatible API server for LLMling agents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from llmling_agent.log import get_logger
from llmling_agent_server.completion_helpers import stream_response
from llmling_agent_server.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    OpenAIMessage,
    OpenAIModelInfo,
)


if TYPE_CHECKING:
    from llmling_agent import AgentPool

logger = get_logger(__name__)


class OpenAIServer:
    """OpenAI-compatible API server backed by LLMling agents."""

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
            logger.exception("Error processing chat completion")
            raise HTTPException(500, f"Error: {e!s}") from e


if __name__ == "__main__":
    import asyncio
    import logging

    import httpx
    import uvicorn

    from llmling_agent import AgentPool

    async def test_client():
        """Test the API with a direct HTTP request."""
        logger = logging.getLogger(__name__)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                headers={"Authorization": "Bearer dummy"},
                json={
                    "model": "gpt-4o-mini",
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
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                print("\n")
            else:
                logger.error("Response error: %s", response.text)

    async def main():
        """Run server and test client."""
        pool = AgentPool[None]()
        await pool.add_agent("gpt-4o-mini", model="openai:gpt-4o-mini")
        async with pool:  # Ensure pool is properly initialized
            server = OpenAIServer(pool)
            config = uvicorn.Config(
                server.app, host="0.0.0.0", port=8000, log_level="info"
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    asyncio.run(main())
