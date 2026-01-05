"""AskUser tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class AskUserTool(Tool[str]):
    """Tool for asking the user clarifying questions.

    Enables agents to ask users for additional information or clarification
    when needed to complete a task effectively.
    """

    def get_callable(self) -> Callable[..., Awaitable[str]]:
        """Get the tool callable."""
        return self._execute

    async def _execute(
        self,
        ctx: AgentContext,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Ask the user a clarifying question.

        Args:
            ctx: Agent execution context.
            prompt: Question to ask the user.
            response_schema: Optional JSON schema for structured response.

        Returns:
            The user's response as a string.
        """
        from mcp.types import ElicitRequestFormParams, ElicitResult, ErrorData

        schema = response_schema or {"type": "string"}
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
