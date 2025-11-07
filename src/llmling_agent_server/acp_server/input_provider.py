"""ACP-based input provider for agent interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp import types

from llmling_agent.log import get_logger
from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from acp.schema import PermissionOption
    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.tools.base import Tool
    from llmling_agent_server.acp_server.session import ACPSession

logger = get_logger(__name__)


def _create_permission_options() -> list[PermissionOption]:
    """Create all 4 permission options for tool confirmation."""
    from acp.schema import PermissionOption

    return [
        PermissionOption(
            option_id="allow-once",
            name="Allow once",
            kind="allow_once",
        ),
        PermissionOption(
            option_id="allow-always",
            name="Allow always",
            kind="allow_always",
        ),
        PermissionOption(
            option_id="reject-once",
            name="Reject once",
            kind="reject_once",
        ),
        PermissionOption(
            option_id="reject-always",
            name="Reject always",
            kind="reject_always",
        ),
    ]


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
        # Track tool approval state: tool_name -> "allow_always" | "reject_always"
        self._tool_approvals: dict[str, str] = {}

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
            # Check if we have a standing approval/rejection for this tool
            if tool.name in self._tool_approvals:
                standing_decision = self._tool_approvals[tool.name]
                if standing_decision == "allow_always":
                    logger.debug("Auto-allowing tool (allow_always)", name=tool.name)
                    return "allow"
                if standing_decision == "reject_always":
                    logger.debug("Auto-rejecting tool (reject_always)", name=tool.name)
                    return "skip"

            # Create a descriptive title for the permission request
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            title = f"Execute tool {tool.name!r} with args: {args_str}"

            # Use a unique tool call ID (could be more sophisticated)
            tool_call_id = f"{tool.name}_{hash(frozenset(args.items()))}"

            # Create all 4 permission options
            options = _create_permission_options()

            # Request permission from the client
            response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=options,
            )

            # Map ACP permission response to our confirmation result
            if response.outcome.outcome == "selected":
                return self._handle_permission_response(
                    response.outcome.option_id, tool.name
                )
            if response.outcome.outcome == "cancelled":
                return "skip"
            # Handle other outcomes
            logger.warning(
                "Unexpected permission outcome", outcome=response.outcome.outcome
            )

        except Exception as e:
            logger.exception("Failed to get tool confirmation", error=e)
            # Default to abort on error to be safe
            return "abort_run"
        else:
            return "abort_run"

    def _handle_permission_response(
        self, option_id: str, tool_name: str
    ) -> ConfirmationResult:
        """Handle permission response and update tool approval state."""
        match option_id:
            case "allow-once":
                return "allow"
            case "allow-always":
                self._tool_approvals[tool_name] = "allow_always"
                logger.info("Tool set to always allow", name=tool_name)
                return "allow"
            case "reject-once":
                return "skip"
            case "reject-always":
                self._tool_approvals[tool_name] = "reject_always"
                logger.info("Tool set to always reject", name=tool_name)
                return "skip"
            case _:
                logger.warning("Unknown permission option", option_id=option_id)
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
            logger.info("Elicitation request", message=params.message)

            tool_call_id = f"elicit_{hash(params.message)}"
            title = f"Elicitation: {params.message}"

            # Use simple options for elicitation (not the full tool approval set)
            from acp.schema import PermissionOption

            options = [
                PermissionOption(
                    option_id="accept",
                    name="Accept",
                    kind="allow_once",
                ),
                PermissionOption(
                    option_id="decline",
                    name="Decline",
                    kind="reject_once",
                ),
            ]

            response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=options,
            )

            # Convert permission response to elicitation result
            if response.outcome.outcome == "selected":
                if response.outcome.option_id == "accept":
                    # For dummy implementation, return acceptance with empty content
                    return types.ElicitResult(action="accept", content={})
                return types.ElicitResult(action="decline")
            if response.outcome.outcome == "cancelled":
                return types.ElicitResult(action="cancel")
            return types.ElicitResult(action="cancel")

        except Exception as e:
            logger.exception("Failed to handle elicitation", error=e)
            return types.ErrorData(
                code=types.INTERNAL_ERROR, message=f"Elicitation failed: {e}"
            )

    def clear_tool_approvals(self) -> None:
        """Clear all stored tool approval decisions.

        This resets the "allow_always" and "reject_always" states,
        so tools will ask for permission again.
        """
        self._tool_approvals.clear()
        logger.info("Cleared all tool approval decisions")

    def get_tool_approval_state(self) -> dict[str, str]:
        """Get current tool approval state for debugging/inspection.

        Returns:
            Dictionary mapping tool names to their approval state
            ("allow_always" or "reject_always")
        """
        return self._tool_approvals.copy()
