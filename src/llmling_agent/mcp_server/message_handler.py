"""FastMCP message handler for llmling-agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from mcp.shared.session import RequestResponder
    import mcp.types

    from llmling_agent.mcp_server.client import MCPClient

logger = get_logger(__name__)


class MCPMessageHandler:
    """Custom message handler that bridges FastMCP to llmling-agent notifications."""

    def __init__(self, client: MCPClient) -> None:
        self.client = client

    async def __call__(
        self,
        message: RequestResponder[mcp.types.ServerRequest, mcp.types.ClientResult]
        | mcp.types.ServerNotification
        | Exception,
    ) -> None:
        """Handle FastMCP messages by dispatching to appropriate handlers."""
        return await self.dispatch(message)

    async def dispatch(
        self,
        message: RequestResponder[mcp.types.ServerRequest, mcp.types.ClientResult]
        | mcp.types.ServerNotification
        | Exception,
    ) -> None:
        """Main dispatch method called by FastMCP."""
        # Handle all messages
        from mcp.shared.session import RequestResponder
        import mcp.types

        await self.on_message(message)
        match message:
            # requests
            case RequestResponder():
                await self.on_request(message)
                # Handle specific requests
                match message.request.root:
                    case mcp.types.PingRequest():
                        await self.on_ping(message.request.root)
                    case mcp.types.ListRootsRequest():
                        await self.on_list_roots(message.request.root)
                    case mcp.types.CreateMessageRequest():
                        await self.on_create_message(message.request.root)

            # notifications
            case mcp.types.ServerNotification():
                await self.on_notification(message)
                # Handle specific notifications
                match message.root:
                    case mcp.types.CancelledNotification():
                        await self.on_cancelled(message.root)
                    case mcp.types.ProgressNotification():
                        await self.on_progress(message.root)
                    case mcp.types.LoggingMessageNotification():
                        await self.on_logging_message(message.root)
                    case mcp.types.ToolListChangedNotification():
                        await self.on_tool_list_changed(message.root)
                    case mcp.types.ResourceListChangedNotification():
                        await self.on_resource_list_changed(message.root)
                    case mcp.types.PromptListChangedNotification():
                        await self.on_prompt_list_changed(message.root)
                    case mcp.types.ResourceUpdatedNotification():
                        await self.on_resource_updated(message.root)

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

    async def on_tool_list_changed(
        self, message: mcp.types.ToolListChangedNotification
    ) -> None:
        """Handle tool list changes by refreshing tools."""
        logger.info("MCP tool list changed: %s", message)
        # Schedule async refresh - use create_task to avoid blocking
        task = asyncio.create_task(self.client._refresh_tools())
        # Store reference to avoid warning about unawaited task
        task.add_done_callback(
            lambda t: t.exception() if t.done() and t.exception() else None
        )

    async def on_resource_list_changed(
        self, message: mcp.types.ResourceListChangedNotification
    ) -> None:
        """Handle resource list changes."""
        logger.info("MCP resource list changed: %s", message)

    async def on_resource_updated(
        self, message: mcp.types.ResourceUpdatedNotification
    ) -> None:
        """Handle resource updates."""
        # ResourceUpdatedNotification has uri directly, not in params
        logger.info("MCP resource updated: %s", getattr(message, "uri", "unknown"))

    async def on_progress(self, message: mcp.types.ProgressNotification) -> None:
        """Handle progress notifications with proper context."""
        # Note: Progress notifications from MCP servers are now handled per-tool-call
        # with the contextual progress handler, so global notifications are ignored

    async def on_prompt_list_changed(
        self, message: mcp.types.PromptListChangedNotification
    ) -> None:
        """Handle prompt list changes."""
        logger.info("MCP prompt list changed: %s", message)

    async def on_cancelled(self, message: mcp.types.CancelledNotification) -> None:
        """Handle cancelled operations."""
        logger.info("MCP operation cancelled: %s", message)

    async def on_logging_message(
        self, message: mcp.types.LoggingMessageNotification
    ) -> None:
        """Handle server log messages."""
        # This is handled by _log_handler, but keep for completeness

    async def on_exception(self, message: Exception) -> None:
        """Handle exceptions."""
        logger.error("MCP client exception: %s", message)

    async def on_ping(self, message: mcp.types.PingRequest) -> None:
        """Handle ping requests."""

    async def on_list_roots(self, message: mcp.types.ListRootsRequest) -> None:
        """Handle list roots requests."""

    async def on_create_message(self, message: mcp.types.CreateMessageRequest) -> None:
        """Handle create message requests."""
