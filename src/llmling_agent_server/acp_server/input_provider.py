"""ACP-based input provider for agent interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp import types

from llmling_agent.log import get_logger
from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.tools.base import Tool
    from llmling_agent_server.acp_server.session import ACPSession

logger = get_logger(__name__)


class ACPInputProvider(InputProvider):
    """Input provider that uses ACP session for user interactions.

    This provider enables tool confirmation and elicitation requests
    through the ACP protocol, allowing clients to interact with agents
    for permission requests and additional input.
    """

    def __init__(self, session: ACPSession) -> None:
        """Initialize ACP input provider.

        Args:
            session: Active ACP session for handling requests
        """
        self.session = session

    async def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        """Get tool execution confirmation via ACP request permission.

        Uses the ACP session's request_permission mechanism to ask
        the client for confirmation before executing the tool.

        Args:
            context: Current agent context
            tool: Information about the tool to be executed
            args: Tool arguments that will be passed to the tool
            message_history: Optional conversation history

        Returns:
            Confirmation result indicating whether to allow, skip, or abort
        """
        try:
            # Create a descriptive title for the permission request
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            title = f"Execute tool '{tool.name}' with args: {args_str}"

            # Use a unique tool call ID (could be more sophisticated)
            tool_call_id = f"{tool.name}_{hash(frozenset(args.items()))}"

            # Request permission from the client
            response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
            )

            # Map ACP permission response to our confirmation result
            if response.outcome.outcome == "selected":
                return "allow"
            if response.outcome.outcome == "cancelled":
                return "skip"
            # Handle other outcomes
            logger.warning(f"Unexpected permission outcome: {response.outcome.outcome}")
            return "abort_run"

        except Exception as e:
            logger.exception(f"Failed to get tool confirmation: {e}")
            # Default to abort on error to be safe
            return "abort_run"

    async def get_elicitation(
        self,
        context: AgentContext[Any],
        params: types.ElicitRequestParams,
        message_history: list[ChatMessage] | None = None,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request.

        This is a dummy implementation for experimentation.
        Uses the same ACP permission mechanism as a placeholder
        for actual elicitation functionality.

        Args:
            context: Current agent context
            params: MCP elicit request parameters
            message_history: Optional conversation history

        Returns:
            Elicit result with user's response or error data
        """
        try:
            # For now, use request_permission as a dummy mechanism
            # In a real implementation, this would use a proper elicitation endpoint
            logger.info(f"Elicitation request: {params.message}")

            tool_call_id = f"elicit_{hash(params.message)}"
            title = f"Elicitation: {params.message}"

            response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
            )

            # Convert permission response to elicitation result
            if response.outcome.outcome == "selected":
                # For dummy implementation, return acceptance with empty content
                return types.ElicitResult(action="accept", content={})
            if response.outcome.outcome == "cancelled":
                return types.ElicitResult(action="decline")
            return types.ElicitResult(action="cancel")

        except Exception as e:
            logger.exception(f"Failed to handle elicitation: {e}")
            return types.ErrorData(
                code=types.INTERNAL_ERROR, message=f"Elicitation failed: {e}"
            )
