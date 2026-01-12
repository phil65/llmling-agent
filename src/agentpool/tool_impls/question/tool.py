"""AskUser tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class QuestionOption:
    """Option for a question in OpenCode format.

    Represents a single choice that can be presented to the user.
    """

    label: str
    """Display text (1-5 words, concise)."""

    description: str
    """Explanation of the choice."""


@dataclass
class OpenCodeQuestionInfo:
    """Question information in OpenCode format.

    This matches OpenCode's QuestionInfo schema used by the TUI.
    """

    question: str
    """Complete question text."""

    header: str
    """Very short label (max 12 chars) - used for tab headers in the TUI."""

    options: list[QuestionOption]
    """Available choices for the user to select from."""

    multiple: bool = False
    """Allow selecting multiple choices."""


@dataclass
class QuestionTool(Tool[str]):
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
                # Content is a dict with "value" key per MCP spec
                if isinstance(content, dict) and "value" in content:
                    value = content["value"]
                    # Handle list responses (multi-select)
                    if isinstance(value, list):
                        return ", ".join(str(v) for v in value)
                    return str(value)
                # Fallback for plain content
                return str(content)
            case ElicitResult(action="cancel"):
                return "User cancelled the request"
            case ElicitResult():
                return "User declined to answer"
            case ErrorData(message=message):
                return f"Error: {message}"
            case _ as unreachable:
                assert_never(unreachable)
