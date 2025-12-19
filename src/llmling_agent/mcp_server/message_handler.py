"""FastMCP message handler for llmling-agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.shared.session import RequestResponder
    import mcp.types

    from llmling_agent.mcp_server import MCPClient

logger = get_logger(__name__)


@dataclass
class MCPMessageHandler:
    """Custom message handler that bridges FastMCP to llmling-agent notifications."""

    client: MCPClient
    """The MCP client instance."""
    tool_change_callback: Callable[[], Awaitable[None]] | None = None
    """Tool change callback."""
    prompt_change_callback: Callable[[], Awaitable[None]] | None = None
    """Prompt change callback."""
    resource_change_callback: Callable[[], Awaitable[None]] | None = None
    """Resource change callback."""

    async def __call__(
        self,
        message: RequestResponder[mcp.types.ServerRequest, mcp.types.ClientResult]
        | mcp.types.ServerNotification
        | Exception,
    ) -> None:
        """Handle FastMCP messages by dispatching to appropriate handlers."""
        from mcp.shared.session import RequestResponder
        import mcp.types

        await self.on_message(message)
        match message:
            # requests
            case RequestResponder() as responder:
                await self.on_request(responder)
                # Handle specific requests
                root = responder.request.root
                match root:
                    case mcp.types.PingRequest():
                        await self.on_ping(root)
                    case mcp.types.ListRootsRequest():
                        await self.on_list_roots(root)
                    case mcp.types.CreateMessageRequest():
                        await self.on_create_message(root)

            case mcp.types.ServerNotification() as notification:
                await self.on_notification(notification)
                root = notification.root
                match root:
                    case mcp.types.CancelledNotification():
                        await self.on_cancelled(root)
                    case mcp.types.ProgressNotification():
                        await self.on_progress(root)
                    case mcp.types.LoggingMessageNotification():
                        await self.on_logging_message(root)
                    case mcp.types.ToolListChangedNotification():
                        await self.on_tool_list_changed(root)
                    case mcp.types.ResourceListChangedNotification():
                        await self.on_resource_list_changed(root)
                    case mcp.types.PromptListChangedNotification():
                        await self.on_prompt_list_changed(root)
                    case mcp.types.ResourceUpdatedNotification():
                        await self.on_resource_updated(root)
                    case mcp.types.ElicitCompleteNotification():
                        await self.on_elicit_complete(root)

            case Exception():
                await self.on_exception(message)

    async def on_message(
        self,
        message: RequestResponder[mcp.types.ServerRequest, mcp.types.ClientResult]
        | mcp.types.ServerNotification
        | Exception,
    ) -> None:
        """Handle generic messages."""

    async def on_request(
        self, message: RequestResponder[mcp.types.ServerRequest, mcp.types.ClientResult]
    ) -> None:
        """Handle requests."""

    async def on_notification(self, message: mcp.types.ServerNotification) -> None:
        """Handle server notifications."""

    async def on_tool_list_changed(self, message: mcp.types.ToolListChangedNotification) -> None:
        """Handle tool list changes."""
        logger.info("MCP tool list changed", message=message)
        # Call the tool change callback if provided
        if self.tool_change_callback:
            await self.tool_change_callback()

    async def on_resource_list_changed(
        self, message: mcp.types.ResourceListChangedNotification
    ) -> None:
        """Handle resource list changes."""
        logger.info("MCP resource list changed", message=message)
        # Call the resource change callback if provided
        if self.resource_change_callback:
            await self.resource_change_callback()

    async def on_resource_updated(self, message: mcp.types.ResourceUpdatedNotification) -> None:
        """Handle resource updates."""
        # ResourceUpdatedNotification has uri directly, not in params
        logger.info("MCP resource updated", uri=getattr(message, "uri", "unknown"))

    async def on_progress(self, message: mcp.types.ProgressNotification) -> None:
        """Handle progress notifications with proper context."""
        # Note: Progress notifications from MCP servers are now handled per-tool-call
        # with the contextual progress handler, so global notifications are ignored

    async def on_prompt_list_changed(
        self, message: mcp.types.PromptListChangedNotification
    ) -> None:
        """Handle prompt list changes."""
        logger.info("MCP prompt list changed", message=message)
        # Call the prompt change callback if provided
        if self.prompt_change_callback:
            await self.prompt_change_callback()

    async def on_cancelled(self, message: mcp.types.CancelledNotification) -> None:
        """Handle cancelled operations."""
        logger.info("MCP operation cancelled", message=message)

    async def on_logging_message(self, message: mcp.types.LoggingMessageNotification) -> None:
        """Handle server log messages."""
        # This is handled by _log_handler, but keep for completeness

    async def on_exception(self, message: Exception) -> None:
        """Handle exceptions."""
        logger.error("MCP client exception", error=message)

    async def on_ping(self, message: mcp.types.PingRequest) -> None:
        """Handle ping requests."""

    async def on_list_roots(self, message: mcp.types.ListRootsRequest) -> None:
        """Handle list roots requests."""

    async def on_create_message(self, message: mcp.types.CreateMessageRequest) -> None:
        """Handle create message requests."""

    async def on_elicit_complete(self, message: mcp.types.ElicitCompleteNotification) -> None:
        """Handle elicitation completion notifications.

        Sent by servers when a URL mode elicitation completes out-of-band.
        """
        logger.info(
            "MCP elicitation completed",
            elicitation_id=message.params.elicitationId,
        )
