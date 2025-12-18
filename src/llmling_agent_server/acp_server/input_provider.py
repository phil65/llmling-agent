"""ACP-based input provider for agent interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import webbrowser

from mcp import types

from acp import AllowedOutcome, PermissionOption
from acp.utils import DEFAULT_PERMISSION_OPTIONS
from llmling_agent.log import get_logger
from llmling_agent.ui.base import InputProvider


if TYPE_CHECKING:
    from acp import RequestPermissionResponse
    from llmling_agent.agents.context import ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools.base import Tool
    from llmling_agent_server.acp_server.session import ACPSession

logger = get_logger(__name__)


def _create_enum_elicitation_options(schema: dict[str, Any]) -> list[PermissionOption] | None:
    """Create permission options for enum elicitation.

    max 4 options (not more ids available)
    """
    enum_values = schema.get("enum", [])
    enum_names = schema.get("enumNames", enum_values)  # Use enumNames if available
    # Limit to 3 choices + cancel to keep ACP UI reasonable
    if len(enum_values) > 3:  # noqa: PLR2004
        logger.warning("Enum truncated for UI", enum_count=len(enum_values), showing_count=3)
        enum_values = enum_values[:3]
        enum_names = enum_names[:3]

    if not enum_values:
        return None

    options = []
    for i, (value, name) in enumerate(zip(enum_values, enum_names, strict=False)):
        opt_id = f"enum_{i}_{value}"
        option = PermissionOption(option_id=opt_id, name=str(name), kind="allow_once")
        options.append(option)
    # Add cancel option
    options.append(PermissionOption(option_id="cancel", name="Cancel", kind="reject_always"))

    return options


def _is_boolean_schema(schema: dict[str, Any]) -> bool:
    """Check if the elicitation schema is requesting a boolean value."""
    # Direct boolean type
    if schema.get("type") == "boolean":
        return True

    # Check if it's an object with a single boolean property (common pattern)
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if len(properties) == 1:
            prop_schema = next(iter(properties.values()))
            return bool(prop_schema.get("type") == "boolean")

    return False


def _is_enum_schema(schema: dict[str, Any]) -> bool:
    """Check if the elicitation schema is requesting an enum value."""
    return schema.get("type") == "string" and "enum" in schema


def _create_boolean_elicitation_options() -> list[PermissionOption]:
    """Create permission options for boolean elicitation (Yes/No)."""
    return [
        PermissionOption(option_id="true", name="Yes", kind="allow_once"),
        PermissionOption(option_id="false", name="No", kind="reject_once"),
        PermissionOption(option_id="cancel", name="Cancel", kind="reject_always"),
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
        self._tool_approvals: dict[str, str] = {}  #  tool_name -> "allow_always" | "reject_always"

    async def get_tool_confirmation(
        self,
        context: NodeContext[Any],
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> ConfirmationResult:
        """Get tool execution confirmation via ACP request permission.

        Uses the ACP session's request_permission mechanism to ask
        the client for confirmation before executing the tool.

        Args:
            context: Current node context
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
                    logger.debug("Auto-allowing tool", tool_name=tool.name, reason="allow_always")
                    return "allow"
                if standing_decision == "reject_always":
                    logger.debug("Auto-rejecting tool", tool_name=tool.name, reason="reject_always")
                    return "skip"

            # Create a descriptive title for the permission request
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            response = await self.session.requests.request_permission(
                tool_call_id=f"{tool.name}_{hash(frozenset(args.items()))}",
                title=f"Execute tool {tool.name!r} with args: {args_str}",
                options=DEFAULT_PERMISSION_OPTIONS,
            )
            # Map ACP permission response to our confirmation result
            if isinstance(response.outcome, AllowedOutcome):
                return self._handle_permission_response(response.outcome.option_id, tool.name)
            if response.outcome.outcome == "cancelled":
                return "skip"
            # Handle other outcomes
            logger.warning("Unexpected permission outcome", outcome=response.outcome.outcome)

        except Exception:
            logger.exception("Failed to get tool confirmation")
            # Default to abort on error to be safe
            return "abort_run"
        else:
            return "abort_run"

    def _handle_permission_response(self, option_id: str, tool_name: str) -> ConfirmationResult:
        """Handle permission response and update tool approval state."""
        match option_id:
            case "allow-once":
                return "allow"
            case "allow-always":
                self._tool_approvals[tool_name] = "allow_always"
                logger.info("Tool approval set", tool_name=tool_name, approval="allow_always")
                return "allow"
            case "reject-once":
                return "skip"
            case "reject-always":
                self._tool_approvals[tool_name] = "reject_always"
                logger.info("Tool approval set", tool_name=tool_name, approval="reject_always")
                return "skip"
            case _:
                logger.warning("Unknown permission option", option_id=option_id)
                return "abort_run"

    async def get_elicitation(  # noqa: PLR0911
        self,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request with basic schema support.

        Currently supports boolean schemas via ACP permission options.
        Other schemas fall back to accept/decline options.

        Args:
            params: MCP elicit request parameters

        Returns:
            Elicit result with user's response or error data
        """
        try:
            # Handle URL mode elicitation (OAuth, credentials, payments)
            if isinstance(params, types.ElicitRequestURLParams):
                msg = "URL elicitation request"
                elicit_id = params.elicitationId
                logger.info(msg, message=params.message, url=params.url, elicitation_id=elicit_id)
                tool_call_id = f"elicit_url_{elicit_id}"
                title = f"URL Authorization: {params.message}"
                url_options = [
                    PermissionOption(option_id="accept", name="Open URL", kind="allow_once"),
                    PermissionOption(option_id="decline", name="Decline", kind="reject_once"),
                ]
                response = await self.session.requests.request_permission(
                    tool_call_id=tool_call_id,
                    title=title,
                    options=url_options,
                )
                if isinstance(response.outcome, AllowedOutcome):
                    if response.outcome.option_id == "accept":
                        webbrowser.open(params.url)
                        return types.ElicitResult(action="accept")
                    return types.ElicitResult(action="decline")
                return types.ElicitResult(action="cancel")

            # Form mode elicitation
            schema = params.requestedSchema
            logger.info("Elicitation request", message=params.message, schema=schema)
            tool_call_id = f"elicit_{hash(params.message)}"
            title = f"Elicitation: {params.message}"

            if _is_boolean_schema(schema):
                options: list[PermissionOption] | None = _create_boolean_elicitation_options()
                response = await self.session.requests.request_permission(
                    tool_call_id=tool_call_id,
                    title=title,
                    options=options,
                )
                return self._handle_boolean_elicitation_response(response, schema)
            if _is_enum_schema(schema) and (options := _create_enum_elicitation_options(schema)):
                response = await self.session.requests.request_permission(
                    tool_call_id=tool_call_id,
                    title=title,
                    options=options,
                )
                return _handle_enum_elicitation_response(response, schema)

            options = [
                PermissionOption(option_id="accept", name="Accept", kind="allow_once"),
                PermissionOption(option_id="decline", name="Decline", kind="reject_once"),
            ]
            response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=options,
            )

            # Convert permission response to elicitation result
            if isinstance(response.outcome, AllowedOutcome):
                if response.outcome.option_id == "accept":
                    # For non-boolean schemas, return empty content
                    return types.ElicitResult(action="accept", content={})
                return types.ElicitResult(action="decline")
            if response.outcome.outcome == "cancelled":
                return types.ElicitResult(action="cancel")
            return types.ElicitResult(action="cancel")

        except Exception as e:
            logger.exception("Failed to handle elicitation")
            return types.ErrorData(code=types.INTERNAL_ERROR, message=f"Elicitation failed: {e}")

    def _handle_boolean_elicitation_response(  # noqa: PLR0911
        self, response: RequestPermissionResponse, schema: dict[str, Any]
    ) -> types.ElicitResult | types.ErrorData:
        """Handle ACP response for boolean elicitation."""
        if isinstance(response.outcome, AllowedOutcome):
            if response.outcome.option_id == "true":
                # Check if we need to wrap in object structure
                if schema.get("type") == "object":
                    properties = schema.get("properties", {})
                    if len(properties) == 1:
                        prop_name = next(iter(properties.keys()))
                        return types.ElicitResult(action="accept", content={prop_name: True})
                return types.ElicitResult(action="accept", content={"value": True})
            if response.outcome.option_id == "false":
                # Check if we need to wrap in object structure
                if schema.get("type") == "object":
                    properties = schema.get("properties", {})
                    if len(properties) == 1:
                        prop_name = next(iter(properties.keys()))
                        return types.ElicitResult(action="accept", content={prop_name: False})
                return types.ElicitResult(action="accept", content={"value": False})
            if response.outcome.option_id == "cancel":
                return types.ElicitResult(action="cancel")

        # Handle cancelled outcome
        if response.outcome.outcome == "cancelled":
            return types.ElicitResult(action="cancel")

        return types.ElicitResult(action="cancel")

    def clear_tool_approvals(self) -> None:
        """Clear all stored tool approval decisions.

        This resets the "allow_always" and "reject_always" states,
        so tools will ask for permission again.
        """
        approval_count = len(self._tool_approvals)
        self._tool_approvals.clear()
        logger.info("Cleared tool approval decisions", count=approval_count)

    def get_tool_approval_state(self) -> dict[str, str]:
        """Get current tool approval state for debugging/inspection.

        Returns:
            Dictionary mapping tool names to their approval state
            ("allow_always" or "reject_always")
        """
        return self._tool_approvals.copy()


def _handle_enum_elicitation_response(
    response: RequestPermissionResponse, schema: dict[str, Any]
) -> types.ElicitResult | types.ErrorData:
    """Handle ACP response for enum elicitation."""
    from mcp import types

    if isinstance(response.outcome, AllowedOutcome):
        option_id = response.outcome.option_id

        if option_id == "cancel":
            return types.ElicitResult(action="cancel")

        # Extract enum value from option_id format: "enum_{index}_{value}"
        if option_id.startswith("enum_"):
            try:
                parts = option_id.split("_", 2)  # Split into ["enum", index, value]
                if len(parts) >= 3:  # noqa: PLR2004
                    enum_index = int(parts[1])
                    enum_values = schema.get("enum", [])
                    if 0 <= enum_index < len(enum_values):
                        selected_value = enum_values[enum_index]
                        return types.ElicitResult(action="accept", content=selected_value)
            except (ValueError, IndexError):
                pass

        # Fallback if parsing fails
        logger.warning("Failed to parse enum option_id", option_id=option_id)
        return types.ElicitResult(action="cancel")

    return types.ElicitResult(action="cancel")
