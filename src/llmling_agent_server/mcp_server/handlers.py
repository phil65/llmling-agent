"""MCP protocol request handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp
from mcp import types

from llmling_agent.log import get_logger
from llmling_agent.mcp_server import constants


if TYPE_CHECKING:
    from pydantic import AnyUrl

    from llmling_agent_server.mcp_server.server import MCPServer


logger = get_logger(__name__)


def register_handlers(llm_server: MCPServer) -> None:  # noqa: PLR0915
    """Register all MCP protocol handlers.

    Args:
        llm_server: LLMLing server instance
    """

    @llm_server.server.set_logging_level()
    async def handle_set_level(level: mcp.LoggingLevel) -> None:
        """Handle logging level changes."""
        try:
            python_level = constants.MCP_TO_LOGGING[level]
            logger.setLevel(python_level)
            data = f"Log level set to {level}"
            await llm_server.current_session.send_log_message(
                level, data, logger=llm_server.name
            )
        except Exception as exc:
            error_data = mcp.ErrorData(
                message="Error setting log level",
                code=types.INTERNAL_ERROR,
                data=str(exc),
            )
            error = mcp.McpError(error_data)
            raise error from exc

    @llm_server.server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """Handle tools/list request."""
        tools = await llm_server.provider.get_tools()
        return [tool.to_mcp_tool() for tool in tools]

    @llm_server.server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[types.TextContent]:
        """Handle tools/call request."""
        arguments = arguments or {}
        # Filter out _meta from arguments
        args = {k: v for k, v in arguments.items() if not k.startswith("_")}

        tools = await llm_server.provider.get_tools()
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            msg = f"Tool not found: {name}"
            raise ValueError(msg)

        try:
            result = await tool.execute(**args)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as exc:
            logger.exception("Tool execution failed", name=name)
            error_msg = f"Tool execution failed: {exc}"
            return [types.TextContent(type="text", text=error_msg)]

    @llm_server.server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        """Handle prompts/list request."""
        return [p.to_mcp_prompt() for p in await llm_server.provider.get_prompts()]

    async def handle_get_prompt(
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> types.GetPromptResult:
        """Handle prompts/get request."""
        from llmling_agent.mcp_server import conversions

        try:
            parts = await llm_server.provider.get_request_parts(name, arguments)
            messages = [msg for p in parts for msg in conversions.to_mcp_messages(p)]
            return types.GetPromptResult(messages=messages, description=name)
        except KeyError as exc:
            error_data = mcp.ErrorData(code=types.INVALID_PARAMS, message=str(exc))
            raise mcp.McpError(error_data) from exc
        except Exception as exc:
            error_data = mcp.ErrorData(code=types.INTERNAL_ERROR, message=str(exc))
            raise mcp.McpError(error_data) from exc

    @llm_server.server.progress_notification()
    async def handle_progress(
        token: str | int,
        progress: float,
        total: float | None,
        message: str | None = None,
    ) -> None:
        """Handle progress notifications from client."""
        logger.debug(
            "Progress notification",
            token=token,
            progress=progress,
            total=total,
            message=message,
        )

    @llm_server.server.subscribe_resource()
    async def handle_subscribe(uri: AnyUrl) -> None:
        """Subscribe to resource updates."""
        uri_str = str(uri)
        llm_server._subscriptions[uri_str].add(llm_server.current_session)
        logger.debug("Added subscription", uri=uri)

    @llm_server.server.unsubscribe_resource()
    async def handle_unsubscribe(uri: AnyUrl) -> None:
        """Unsubscribe from resource updates."""
        if (uri_str := str(uri)) in llm_server._subscriptions:
            llm_server._subscriptions[uri_str].discard(llm_server.current_session)
            if not llm_server._subscriptions[uri_str]:
                del llm_server._subscriptions[uri_str]
            msg = "Removed subscription"
            logger.debug(msg, uri=uri, session=llm_server.current_session)
