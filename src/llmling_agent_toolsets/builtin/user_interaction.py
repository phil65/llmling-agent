"""Provider for user interaction tools."""

from __future__ import annotations

from typing import Any, assert_never

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import StaticResourceProvider


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
    from mcp.types import ElicitRequestFormParams, ElicitResult, ErrorData

    schema = response_schema or {"type": "string"}  # string schema if no none provided
    params = ElicitRequestFormParams(message=prompt, requestedSchema=schema)
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
        case _ as unreachable:
            assert_never(unreachable)


class UserInteractionTools(StaticResourceProvider):
    """Provider for user interaction tools."""

    def __init__(self, name: str = "user_interaction") -> None:
        super().__init__(name=name)
        self._tools = [self.create_tool(ask_user, category="other", open_world=True)]
