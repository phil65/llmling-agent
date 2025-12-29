"""OpenCode-based input provider for agent interactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp import types

from agentpool.log import get_logger
from agentpool.ui.base import InputProvider


if TYPE_CHECKING:
    from agentpool.agents.context import ConfirmationResult
    from agentpool.messaging import ChatMessage
    from agentpool.messaging.context import NodeContext
    from agentpool.tools.base import Tool
    from agentpool_server.opencode_server.state import ServerState

logger = get_logger(__name__)

# OpenCode permission responses
PermissionResponse = str  # "once" | "always" | "reject"


@dataclass
class PendingPermission:
    """A pending permission request awaiting user response."""

    permission_id: str
    tool_name: str
    args: dict[str, Any]
    future: asyncio.Future[PermissionResponse]
    created_at: float = field(default_factory=lambda: __import__("time").time())


class OpenCodeInputProvider(InputProvider):
    """Input provider that uses OpenCode SSE/REST for user interactions.

    This provider enables tool confirmation and elicitation requests
    through the OpenCode protocol. When a tool needs confirmation:
    1. A permission request is created and stored
    2. An SSE event is broadcast to notify clients
    3. The provider awaits a response via the REST endpoint
    4. The client POSTs to /session/{id}/permissions/{permissionID} to respond
    """

    def __init__(self, state: ServerState, session_id: str) -> None:
        """Initialize OpenCode input provider.

        Args:
            state: Server state for broadcasting events
            session_id: The session ID for this provider
        """
        self.state = state
        self.session_id = session_id
        self._pending_permissions: dict[str, PendingPermission] = {}
        self._tool_approvals: dict[str, str] = {}  # tool_name -> "always" | "reject"
        self._id_counter = 0

    def _generate_permission_id(self) -> str:
        """Generate a unique permission ID."""
        self._id_counter += 1
        return f"perm_{self._id_counter}_{int(__import__('time').time() * 1000)}"

    async def get_tool_confirmation(
        self,
        context: NodeContext[Any],
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> ConfirmationResult:
        """Get tool execution confirmation via OpenCode permission request.

        Creates a pending permission, broadcasts an SSE event, and waits
        for the client to respond via POST /session/{id}/permissions/{permissionID}.

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
                if standing_decision == "always":
                    logger.debug("Auto-allowing tool", tool_name=tool.name, reason="always")
                    return "allow"
                if standing_decision == "reject":
                    logger.debug("Auto-rejecting tool", tool_name=tool.name, reason="reject")
                    return "skip"

            # Create a pending permission request
            permission_id = self._generate_permission_id()
            future: asyncio.Future[PermissionResponse] = asyncio.get_event_loop().create_future()

            pending = PendingPermission(
                permission_id=permission_id,
                tool_name=tool.name,
                args=args,
                future=future,
            )
            self._pending_permissions[permission_id] = pending

            # Broadcast SSE event to notify clients
            from agentpool_server.opencode_server.models.events import (
                PermissionRequestEvent,
                PermissionRequestProperties,
            )

            max_preview_args = 3
            args_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:max_preview_args])
            if len(args) > max_preview_args:
                args_preview += ", ..."

            event = PermissionRequestEvent(
                properties=PermissionRequestProperties(
                    session_id=self.session_id,
                    permission_id=permission_id,
                    tool_name=tool.name,
                    args_preview=args_preview,
                    message=f"Tool '{tool.name}' wants to execute with args: {args_preview}",
                )
            )
            await self.state.broadcast_event(event)

            logger.info(
                "Permission requested",
                permission_id=permission_id,
                tool_name=tool.name,
            )

            # Wait for the client to respond
            try:
                response = await future
            except asyncio.CancelledError:
                logger.warning("Permission request cancelled", permission_id=permission_id)
                return "skip"
            finally:
                # Clean up the pending permission
                self._pending_permissions.pop(permission_id, None)

            # Map OpenCode response to our confirmation result
            return self._handle_permission_response(response, tool.name)

        except Exception:
            logger.exception("Failed to get tool confirmation")
            return "abort_run"

    def _handle_permission_response(
        self, response: PermissionResponse, tool_name: str
    ) -> ConfirmationResult:
        """Handle permission response and update tool approval state."""
        match response:
            case "once":
                return "allow"
            case "always":
                self._tool_approvals[tool_name] = "always"
                logger.info("Tool approval set", tool_name=tool_name, approval="always")
                return "allow"
            case "reject":
                return "skip"
            case _:
                logger.warning("Unknown permission response", response=response)
                return "abort_run"

    def resolve_permission(self, permission_id: str, response: PermissionResponse) -> bool:
        """Resolve a pending permission request.

        Called by the REST endpoint when the client responds.

        Args:
            permission_id: The permission request ID
            response: The client's response ("once", "always", or "reject")

        Returns:
            True if the permission was found and resolved, False otherwise
        """
        pending = self._pending_permissions.get(permission_id)
        if pending is None:
            logger.warning("Permission not found", permission_id=permission_id)
            return False

        if pending.future.done():
            logger.warning("Permission already resolved", permission_id=permission_id)
            return False

        pending.future.set_result(response)
        logger.info(
            "Permission resolved",
            permission_id=permission_id,
            response=response,
        )
        return True

    def get_pending_permissions(self) -> list[dict[str, Any]]:
        """Get all pending permission requests.

        Returns:
            List of pending permission info dicts
        """
        return [
            {
                "permission_id": p.permission_id,
                "tool_name": p.tool_name,
                "args": p.args,
                "created_at": p.created_at,
            }
            for p in self._pending_permissions.values()
        ]

    async def get_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request.

        For now, this returns a basic decline since OpenCode doesn't have
        a full elicitation UI like ACP. Future versions could add support
        for more complex elicitation flows.

        Args:
            params: MCP elicit request parameters

        Returns:
            Elicit result with user's response or error data
        """
        # For URL elicitation, we could open the URL
        if isinstance(params, types.ElicitRequestURLParams):
            logger.info(
                "URL elicitation request",
                message=params.message,
                url=params.url,
            )
            # Could potentially open URL in browser here
            return types.ElicitResult(action="decline")

        # For form elicitation, we don't have UI support yet
        logger.info(
            "Form elicitation request (not supported)",
            message=params.message,
            schema=getattr(params, "requestedSchema", None),
        )
        return types.ElicitResult(action="decline")

    def clear_tool_approvals(self) -> None:
        """Clear all stored tool approval decisions."""
        approval_count = len(self._tool_approvals)
        self._tool_approvals.clear()
        logger.info("Cleared tool approval decisions", count=approval_count)

    def cancel_all_pending(self) -> int:
        """Cancel all pending permission requests.

        Returns:
            Number of permissions cancelled
        """
        count = 0
        for pending in list(self._pending_permissions.values()):
            if not pending.future.done():
                pending.future.cancel()
                count += 1
        self._pending_permissions.clear()
        logger.info("Cancelled all pending permissions", count=count)
        return count
