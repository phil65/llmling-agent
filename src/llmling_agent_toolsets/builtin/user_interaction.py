"""Provider for user interaction tools."""

from __future__ import annotations

from typing import Any

from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


async def ask_user(  # noqa: D417
    ctx: AgentContext,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Allow LLM to ask user a clarifying question during processing.

    This tool enables agents to ask users for additional information or clarification
    when needed to complete a task effectively.

    Args:
        prompt: Question to ask the user
        response_schema: Optional JSON schema for structured response (defaults to string)

    Returns:
        The user's response as a string
    """
    from mcp.types import ElicitRequestParams, ElicitResult, ErrorData

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    schema = response_schema or {"type": "string"}  # string schema if no none provided
    params = ElicitRequestParams(message=prompt, requestedSchema=schema)
    result = await ctx.handle_elicitation(params)

    match result:
        case ElicitResult(action="accept", content=content):
            return str(content)
        case ElicitResult(action="cancel"):
            return "User cancelled the request"
        case ElicitResult():
            return "User declined to answer"
        case ErrorData(message=message):
            return f"Error: {message}"
        case _:
            return "Unknown error occurred"


def create_user_interaction_tools() -> list[Tool]:
    """Create tools for user interaction operations."""
    return [Tool.from_callable(ask_user, source="builtin", category="other")]


class UserInteractionTools(StaticResourceProvider):
    """Provider for user interaction tools."""

    def __init__(self, name: str = "user_interaction"):
        super().__init__(name=name, tools=create_user_interaction_tools())
