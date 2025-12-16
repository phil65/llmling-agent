"""OpenAI-compatible responses endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from llmling_agent_server.openai_api_server.responses.models import (
    Response,
    ResponseMessage,
    ResponseOutputText,
    ResponseToolCall,
    ResponseUsage,
)


if TYPE_CHECKING:
    from llmling_agent.agents import Agent
    from llmling_agent.agents.acp_agent import ACPAgent
    from llmling_agent_server.openai_api_server.responses.models import ResponseRequest


async def handle_request(request: ResponseRequest, agent: Agent[Any, Any] | ACPAgent) -> Response:
    from fastapi import HTTPException

    match request.input:
        case str():
            content = request.input
        case list():
            # Get last text content from structured input
            last = request.input[-1]["content"]
            text_parts = [p["text"] for p in last if p["type"] == "input_text"]
            content = "\n".join(text_parts)
        case _:
            raise HTTPException(400, "Invalid input format")

    message = await agent.run(content)
    text = ResponseOutputText(text=str(message.content))
    output_msg_id = f"msg_{uuid4().hex}"
    output_msg = ResponseMessage(id=output_msg_id, role="assistant", content=[text])
    output: list[ResponseMessage | ResponseToolCall] = [output_msg]

    calls = [
        ResponseToolCall(type=f"{tc.tool_name}_call", id=tc.tool_call_id)
        for tc in message.get_tool_calls()
    ]
    output = calls + output  # type: ignore

    usage_info: ResponseUsage | None = None
    if message.cost_info and (token_usage := message.cost_info.token_usage):
        # Map the keys correctly from agent's dict to ResponseUsage TypedDict
        input_tk = token_usage.input_tokens
        output_tk = token_usage.output_tokens
        total_tk = token_usage.total_tokens

        usage_info = ResponseUsage(
            input_tokens=input_tk,
            input_tokens_details={},
            output_tokens=output_tk,
            output_tokens_details={},
            total_tokens=total_tk,
        )

    return Response(
        model=request.model,
        output=output,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        tools=request.tools,
        tool_choice=request.tool_choice,
        usage=usage_info,
        metadata=request.metadata,
    )
