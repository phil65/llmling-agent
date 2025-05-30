"""Helpers for OpenAI-compatible API server."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import anyenv

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llmling_agent import AnyAgent
    from llmling_agent_server.models import ChatCompletionRequest

logger = get_logger(__name__)


async def stream_response(
    agent: AnyAgent[Any, Any],
    content: str,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    response_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    try:
        # First chunk with role
        choice = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [choice],
        }
        yield f"data: {anyenv.dump_json(first_chunk)}\n\n"
        async with agent.run_stream(content) as stream:
            async for chunk in stream.stream_text(delta=True):
                # Skip empty chunks
                if not chunk:
                    continue
                choice = {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [choice],
                }
                yield f"data: {anyenv.dump_json(chunk_data)}\n\n"
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {anyenv.dump_json(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error during streaming response")
        choice = {
            "index": 0,
            "delta": {"content": f"Error: {e!s}"},
            "finish_reason": "error",
        }
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [choice],
        }
        yield f"data: {anyenv.dump_json(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
