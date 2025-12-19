"""Command for serving agents via Vercel AI protocol."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent import ChatMessage


logger = get_logger(__name__)


def vercel_command(  # noqa: PLR0915
    ctx: t.Context,
    config: Annotated[str | None, t.Argument(help="Path to agent configuration")] = None,
    agent_name: Annotated[
        str | None, t.Option("--agent", "-a", help="Specific agent to serve")
    ] = None,
    host: Annotated[str, t.Option(help="Host to bind server to")] = "localhost",
    port: Annotated[int, t.Option(help="Port to listen on")] = 8000,
    cors: Annotated[bool, t.Option(help="Enable CORS")] = True,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity")
    ] = False,
) -> None:
    """Serve agents via Vercel AI Data Stream Protocol.

    This creates a server compatible with Vercel AI SDK frontends,
    allowing you to use your agents with Vercel AI UI components.

    The server exposes a POST /chat endpoint that accepts Vercel AI
    protocol requests and streams responses back.

    If --agent is specified, only that agent is served. Otherwise,
    the endpoint accepts an 'agent' field in the request to select
    which agent to use.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic_ai import PartDeltaEvent, PartStartEvent
    from pydantic_ai.messages import TextPart, TextPartDelta
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    import uvicorn

    from llmling_agent import AgentPool, AgentsManifest
    from llmling_agent.agents.events import StreamCompleteEvent

    def on_message(message: ChatMessage[Any]) -> None:
        print(message.format(style="simple"))

    try:
        config_path = resolve_agent_config(config)
    except ValueError as e:
        msg = str(e)
        raise t.BadParameter(msg) from e

    manifest = AgentsManifest.from_file(config_path)
    pool = AgentPool(manifest)

    if show_messages:
        for agent in pool.agents.values():
            agent.message_sent.connect(on_message)

    # Create FastAPI app
    app = FastAPI(
        title="LLMling Agent - Vercel AI Server",
        description="Vercel AI Data Stream Protocol server for LLMling agents",
    )

    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.on_event("startup")
    async def startup() -> None:
        await pool.__aenter__()
        logger.info("Agent pool initialized")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await pool.__aexit__(None, None, None)
        logger.info("Agent pool shut down")

    @app.post("/chat")
    async def chat(request: Request) -> Response:
        """Handle Vercel AI protocol chat requests.

        Implements the Vercel AI Data Stream Protocol:
        https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
        """
        body = await request.body()
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

        # Extract messages from the request
        messages = data.get("messages", [])
        if not messages:
            return JSONResponse({"error": "No messages provided"}, status_code=400)

        # Get the last user message
        last_message = messages[-1]
        user_text = ""
        if last_message.get("role") == "user":
            parts = last_message.get("parts", [])
            for part in parts:
                if part.get("type") == "text":
                    user_text = part.get("text", "")
                    break

        if not user_text:
            return JSONResponse({"error": "No user text found"}, status_code=400)

        # Determine which agent to use
        if agent_name:
            selected_agent = pool.get_agent(agent_name)
        else:
            if not pool.agents:
                return JSONResponse({"error": "No agents available"}, status_code=500)
            first_name = next(iter(pool.agents.keys()))
            selected_agent = pool.get_agent(first_name)

        async def generate_stream() -> AsyncIterator[str]:
            """Generate Vercel AI Data Stream Protocol events.

            Protocol format:
            - Text: 0:"text content"
            - Finish: e:{"finishReason":"stop",...}
            - Done: d:{"finishReason":"stop"}
            """
            try:
                async for event in selected_agent.run_stream(user_text):
                    # Handle pydantic-ai streaming events
                    if isinstance(event, PartStartEvent):
                        # New part started - if it's text, emit it
                        if isinstance(event.part, TextPart) and event.part.content:
                            text = event.part.content
                            escaped = json.dumps(text)
                            yield f"0:{escaped}\n"

                    elif isinstance(event, PartDeltaEvent):
                        # Delta update - emit text deltas
                        if isinstance(event.delta, TextPartDelta):
                            text = event.delta.content_delta
                            if text:
                                escaped = json.dumps(text)
                                yield f"0:{escaped}\n"

                    elif isinstance(event, StreamCompleteEvent):
                        # Stream complete - we've received the final message
                        # The content has already been streamed via deltas
                        pass

                # Send finish event
                finish_data = {
                    "finishReason": "stop",
                    "usage": {"promptTokens": 0, "completionTokens": 0},
                }
                yield f"e:{json.dumps(finish_data)}\n"

                # Send done marker
                done_data = {"finishReason": "stop"}
                yield f"d:{json.dumps(done_data)}\n"

            except Exception as e:
                logger.exception("Error during streaming")
                # Send error as text
                error_msg = f"Error: {e}"
                escaped = json.dumps(error_msg)
                yield f"0:{escaped}\n"
                # Still send finish
                finish_data = {"finishReason": "error"}
                yield f"e:{json.dumps(finish_data)}\n"
                done_data = {"finishReason": "error"}
                yield f"d:{json.dumps(done_data)}\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain; charset=utf-8",
            headers={
                "X-Vercel-AI-Data-Stream": "v1",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/agents")
    async def list_agents() -> dict[str, Any]:
        """List available agents."""
        return {
            "agents": [
                {"name": name, "description": agent.description}
                for name, agent in pool.agents.items()
            ]
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Get log level from the global context
    log_level = ctx.obj.get("log_level", "info") if ctx.obj else "info"

    print(f"Starting Vercel AI server on http://{host}:{port}")
    print(f"Chat endpoint: POST http://{host}:{port}/chat")
    print(f"Available agents: {list(pool.agents.keys())}")

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    import typer

    typer.run(vercel_command)
