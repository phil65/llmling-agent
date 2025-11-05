"""Base input provider class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from mcp import types
    from pydantic import BaseModel

    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.tools.base import Tool


class InputProvider(ABC):
    """Base class for handling all UI interactions."""

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        output_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Any:
        """Get normal input (used by HumanProvider).

        Args:
            context: Current agent context
            prompt: The prompt to show to the user
            output_type: Optional type for structured responses
            message_history: Optional conversation history
        """
        if output_type:
            return await self.get_structured_input(
                context, prompt, output_type, message_history
            )
        return await self.get_text_input(context, prompt, message_history)

    async def get_text_input(
        self,
        context: AgentContext[Any],
        prompt: str,
        message_history: list[ChatMessage] | None = None,
    ) -> str:
        """Get normal text input."""
        raise NotImplementedError

    async def get_structured_input(
        self,
        context: AgentContext[Any],
        prompt: str,
        output_type: type[BaseModel],
        message_history: list[ChatMessage] | None = None,
    ) -> BaseModel:
        """Get structured input, with promptantic and fallback handling."""
        raise NotImplementedError

    @abstractmethod
    def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> Coroutine[Any, Any, ConfirmationResult]:
        """Get tool execution confirmation.

        Args:
            context: Current agent context
            tool: Information about the tool to be executed
            args: Tool arguments
            message_history: Optional conversation history
        """

    @abstractmethod
    def get_elicitation(
        self,
        context: AgentContext[Any],
        params: types.ElicitRequestParams,
        message_history: list[ChatMessage] | None = None,
    ) -> Coroutine[Any, Any, types.ElicitResult | types.ErrorData]:
        """Get user response to elicitation request.

        Args:
            context: Current agent context
            params: MCP elicit request parameters
            message_history: Optional conversation history
        """

    @abstractmethod
    def get_code_input(
        self,
        context: AgentContext[Any],
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> Coroutine[Any, Any, str]:
        """Get multi-line code input.

        Args:
            context: Current agent context
            template: Optional template code
            language: Programming language (for syntax highlighting)
            description: Optional description of expected code
        """
