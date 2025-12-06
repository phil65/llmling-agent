"""Helpers for OpenAI-compatible API server."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import anyenv
from pydantic_ai import PartDeltaEvent, TextPartDelta

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llmling_agent.common_types import SupportsRunStream
    from llmling_agent_server.openai_api_server.completions.models import ChatCompletionRequest

logger = get_logger(__name__)


async def stream_response(
    agent: SupportsRunStream[Any],
    content: str,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str]:
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
        async for event in agent.run_stream(content):
            match event:
                case PartDeltaEvent(delta=TextPartDelta(content_delta=chunk)):
                    # Skip empty chunks
                    if not chunk:
                        continue
                    delta = {"content": chunk}
                    choice = {"index": 0, "delta": delta, "finish_reason": None}
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
        delta = {"content": f"Error: {e!s}"}
        choice = {"index": 0, "delta": delta, "finish_reason": "error"}
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [choice],
        }
        yield f"data: {anyenv.dump_json(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
