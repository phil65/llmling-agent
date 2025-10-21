"""FastMCP-based client implementation for LLMling agent.

This module provides a client for communicating with MCP servers using FastMCP.
It includes support for contextual progress handlers that extend FastMCP's
standard progress callbacks with tool execution context (tool name, call ID, and input).

The key innovation is the signature injection system that allows MCP tools to work
seamlessly with PydanticAI's RunContext while providing rich progress information.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, Self

from pydantic_ai import RunContext
from schemez.functionschema import FunctionSchema

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, Protocol

    import fastmcp
    from fastmcp.client import ClientTransport
    from fastmcp.client.client import ProgressHandler
    from fastmcp.client.elicitation import ElicitationHandler, ElicitResult
    from fastmcp.client.logging import LogMessage
    from fastmcp.client.messages import MessageHandler, MessageHandlerT
    from fastmcp.client.sampling import ClientSamplingHandler
    import mcp
    from mcp.client.session import RequestContext
    from mcp.types import (
        CreateMessageRequestParams,
        ElicitRequestParams,
        Prompt as MCPPrompt,
        Resource as MCPResource,
        SamplingMessage,
        Tool as MCPTool,
    )

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig

    class ContextualProgressHandler(Protocol):
        """Progress handler that includes tool execution context."""

        async def __call__(
            self,
            progress: float,
            total: float | None,
            message: str | None,
            tool_name: str | None = None,
            tool_call_id: str | None = None,
            tool_input: dict[str, Any] | None = None,
        ) -> None: ...


logger = get_logger(__name__)


def extract_tool_call_args(messages, target_tool_call_id: str) -> dict[str, Any]:
    """Extract tool call arguments from message history by tool_call_id.

    Args:
        messages: List of ModelMessage objects from RunContext
        target_tool_call_id: The tool call ID to find arguments for

    Returns:
        Dictionary of tool call arguments, empty dict if not found
    """
    # Import here to avoid circular imports
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    for message in reversed(messages):  # Search from most recent
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if (
                    isinstance(part, ToolCallPart)
                    and part.tool_call_id == target_tool_call_id
                ):
                    return part.args_as_dict()
    return {}


LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}


def _create_tool_signature_with_context(base_signature, tool_name: str) -> Any:
    """Create a function signature that includes RunContext as first parameter.

    This is crucial for PydanticAI integration - it expects tools that need RunContext
    to have it as the first parameter with proper annotation. Without this, PydanticAI
    won't pass the RunContext and we lose access to tool_call_id and other context.

    Args:
        base_signature: Original signature from MCP tool schema (tool parameters only)
        tool_name: Name of the tool (used for documentation)

    Returns:
        New signature: (ctx: RunContext, ...original_params) -> ReturnType

    Example:
        Original: (message: str) -> str
        Result:   (ctx: RunContext, message: str) -> str
    """
    import inspect

    # Create RunContext parameter
    ctx_param = inspect.Parameter(
        "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext
    )

    # Combine with tool parameters
    tool_params = list(base_signature.parameters.values())
    new_params = [ctx_param, *tool_params]

    return base_signature.replace(parameters=new_params)


def _create_tool_annotations_with_context(
    base_annotations: dict[str, Any],
) -> dict[str, Any]:
    """Create function annotations that include RunContext for first parameter.

    Complements _create_tool_signature_with_context by ensuring the annotations
    match the signature. This is required for proper type checking and PydanticAI's
    introspection of tool functions.

    Args:
        base_annotations: Original annotations from MCP tool schema

    Returns:
        New annotations dict with 'ctx': RunContext added to base annotations

    Example:
        Original: {'message': str, 'return': str}
        Result:   {'ctx': RunContext, 'message': str, 'return': str}
    """
    new_annotations = base_annotations.copy()
    new_annotations["ctx"] = RunContext
    return new_annotations


def mcp_tool_to_fn_schema(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function schema format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
    }


class MCPClient:
    """FastMCP-based client for communicating with MCP servers."""

    def __init__(
        self,
        elicitation_callback: ElicitationHandler | None = None,
        sampling_callback: ClientSamplingHandler | None = None,
        progress_handler: ContextualProgressHandler | None = None,
        message_handler: MessageHandlerT | MessageHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        self._elicitation_callback = elicitation_callback
        self._sampling_callback = sampling_callback
        self._contextual_progress_handler = progress_handler
        # Store message handler or mark for lazy creation
        self._message_handler = message_handler
        self._accessible_roots = accessible_roots or []
        self._client: fastmcp.Client | None = None
        self._available_tools: list[MCPTool] = []
        self._connected = False

    async def __aenter__(self) -> Self:
        """Enter context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit context manager and cleanup."""
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("Error during FastMCP client cleanup: %s", e)
            finally:
                self._client = None
                self._connected = False
                self._available_tools = []

    async def _log_handler(self, message: LogMessage) -> None:
        """Handle server log messages."""
        msg = message.data.get("msg", "")
        level = LEVEL_MAP.get(message.level.lower(), logging.INFO)
        logger.log(level, "MCP Server: %s", msg)

    async def _progress_handler_impl(
        self, progress: float, total: float | None, message: str | None
    ):
        """Handle progress updates from server."""
        # This is the global FastMCP progress handler - we don't have tool context here
        # Tool-specific progress handlers are created per call in call_tool()

    async def _elicitation_handler_impl(
        self,
        message: str,
        response_type: type,
        params: ElicitRequestParams,
        context: RequestContext,
    ) -> ElicitResult[dict[str, Any]] | dict[str, Any] | None:
        """Handle elicitation requests from server."""
        if not self._elicitation_callback:
            return None

        try:
            # Direct FastMCP callback - no compatibility layer
            return await self._elicitation_callback(
                message, response_type, params, context
            )
        except Exception:
            logger.exception("Elicitation handler failed")
            from fastmcp.client.elicitation import ElicitResult

            return ElicitResult(action="cancel")

    async def _sampling_handler_impl(
        self,
        messages: list[SamplingMessage],
        params: CreateMessageRequestParams,
        context: RequestContext,
    ) -> str:
        """Handle sampling requests from server."""
        if not self._sampling_callback:
            return "Sampling not supported"

        try:
            result = self._sampling_callback(messages, params, context)
            if inspect.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            logger.exception("Sampling handler failed")
            return f"Sampling failed: {e}"

    async def connect(self, config: MCPServerConfig):
        """Connect to an MCP server using FastMCP.

        Args:
            config: MCP server configuration object
        """
        try:
            # First attempt with configured auth
            self._client = self._get_client(config)
            await self._client.__aenter__()

        except Exception as first_error:
            # OAuth fallback for HTTP/SSE if not already using OAuth
            if _should_try_oauth_fallback(config):
                try:
                    if self._client:
                        with contextlib.suppress(Exception):
                            await self._client.__aexit__(None, None, None)

                    self._client = self._get_client(config, force_oauth=True)
                    await self._client.__aenter__()
                    logger.info("Connected with OAuth fallback")
                except Exception:  # noqa: BLE001
                    raise first_error from None
            else:
                raise

        self._connected = True
        await self._refresh_tools()

    def _get_client(self, config: MCPServerConfig, force_oauth: bool = False):
        """Create FastMCP client based on config."""
        import fastmcp
        from fastmcp.client import SSETransport, StreamableHttpTransport
        from fastmcp.client.transports import StdioTransport

        from llmling_agent_config.mcp_server import (
            SSEMCPServerConfig,
            StdioMCPServerConfig,
            StreamableHTTPMCPServerConfig,
        )

        transport: ClientTransport
        # Create transport based on config type
        match config:
            case StdioMCPServerConfig(command=command, args=args):
                env = config.get_env_vars()
                transport = StdioTransport(command=command, args=args, env=env)
                oauth = False
                if force_oauth:
                    msg = "OAuth is not supported for StdioMCPServerConfig"
                    raise ValueError(msg)

            case SSEMCPServerConfig(url=url, headers=headers, auth=auth):
                transport = SSETransport(url=url, headers=headers)
                oauth = auth.oauth

            case StreamableHTTPMCPServerConfig(url=url, headers=headers, auth=auth):
                transport = StreamableHttpTransport(url=url, headers=headers)
                oauth = auth.oauth
            case _:
                msg = f"Unsupported server config type: {type(config)}"
                raise ValueError(msg)

        # Create message handler if needed
        msg_handler: MessageHandlerT | MessageHandler | None
        if not self._message_handler:
            from llmling_agent.mcp_server.message_handler import MCPMessageHandler

            msg_handler = MCPMessageHandler(self)
        else:
            msg_handler = self._message_handler

        return fastmcp.Client(
            transport,
            log_handler=self._log_handler,
            roots=self._accessible_roots,
            timeout=config.timeout,
            progress_handler=self._progress_handler_impl,
            elicitation_handler=self._elicitation_handler_impl,
            sampling_handler=self._sampling_handler_impl,
            message_handler=msg_handler,
            auth="oauth" if (force_oauth or oauth) else None,
        )

    async def _refresh_tools(self) -> None:
        """Refresh the list of available tools from the server."""
        if not self._client or not self._connected:
            return

        try:
            tools = await self._client.list_tools()
            self._available_tools = tools
            logger.debug("Refreshed %d tools from MCP server", len(tools))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to refresh tools: %s", e)
            self._available_tools = []

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function format."""
        return [
            {"type": "function", "function": mcp_tool_to_fn_schema(tool)}
            for tool in self._available_tools
        ]

    async def list_prompts(self) -> list[MCPPrompt]:
        """Get available prompts from the server."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.list_prompts()
        except Exception as e:
            msg = f"Failed to list prompts: {e}"
            raise RuntimeError(msg) from e

    async def list_resources(self) -> list[MCPResource]:
        """Get available resources from the server."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.list_resources()
        except Exception as e:
            msg = f"Failed to list resources: {e}"
            raise RuntimeError(msg) from e

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> mcp.types.GetPromptResult:
        """Get a specific prompt's content."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.get_prompt_mcp(name, arguments)
        except Exception as e:
            msg = f"Failed to get prompt '{name}': {e}"
            raise RuntimeError(msg) from e

    def convert_tool(self, tool: MCPTool) -> Tool:
        """Create a properly typed callable from MCP tool schema."""
        from llmling_agent import Tool

        schema = mcp_tool_to_fn_schema(tool)
        fn_schema = FunctionSchema.from_dict(schema)
        sig = fn_schema.to_python_signature()

        async def tool_callable(ctx: RunContext, **kwargs: Any) -> str:
            """Dynamically generated MCP tool wrapper."""
            # Filter out None values for optional params
            schema_props = tool.inputSchema.get("properties", {})
            required_props = set(tool.inputSchema.get("required", []))

            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in required_props or (k in schema_props and v is not None)
            }
            tc_id = ctx.tool_call_id if ctx else None

            return await self.call_tool(
                tool.name,
                filtered_kwargs,
                tool_call_id=tc_id,
                run_context=ctx,
            )

        # Set proper signature and annotations with RunContext support
        tool_callable.__signature__ = _create_tool_signature_with_context(sig, tool.name)  # type: ignore
        tool_callable.__annotations__ = _create_tool_annotations_with_context(
            fn_schema.get_annotations()
        )

        tool_callable.__name__ = tool.name
        tool_callable.__doc__ = tool.description or "No description provided."
        meta = {"mcp_tool": tool.name}
        return Tool.from_callable(tool_callable, source="mcp", metadata=meta)

    def _create_progress_handler_with_context(
        self, tool_name: str, tool_call_id: str, tool_input: dict[str, Any]
    ) -> ProgressHandler:
        """Create a FastMCP-compatible progress handler with baked-in context."""

        async def fastmcp_progress_handler(
            progress: float, total: float | None, message: str | None
        ):
            if self._contextual_progress_handler:
                await self._contextual_progress_handler(
                    progress, total, message, tool_name, tool_call_id, tool_input
                )

        return fastmcp_progress_handler

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
        tool_call_id: str | None = None,
        run_context: RunContext | None = None,
    ) -> str:
        """Call an MCP tool."""
        from mcp.types import TextContent

        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            # Create progress handler if we have handler
            progress_handler = None
            if self._contextual_progress_handler:
                if run_context and run_context.tool_call_id and run_context.tool_name:
                    # Extract tool args from message history
                    tool_input = extract_tool_call_args(
                        run_context.messages, run_context.tool_call_id
                    )
                    progress_handler = self._create_progress_handler_with_context(
                        run_context.tool_name, run_context.tool_call_id, tool_input
                    )
                else:
                    # Fallback to using passed arguments (direct tool call)
                    progress_handler = self._create_progress_handler_with_context(
                        name, tool_call_id or "", arguments or {}
                    )

            # Use FastMCP's call_tool method with optional progress handler
            result = await self._client.call_tool(
                name, arguments or {}, progress_handler=progress_handler
            )

            # FastMCP returns a CallToolResult with structured data
            # For compatibility, return text content
            if result.content:
                if isinstance(result.content[0], TextContent):
                    return result.content[0].text
                # TODO: proper support.
                return str(result.content)
            if result.data is not None:
                return str(result.data)
        except Exception as e:
            msg = f"MCP tool call failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return "Tool executed successfully"


def _should_try_oauth_fallback(config: MCPServerConfig) -> bool:
    """Check if OAuth fallback should be attempted."""
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    # No fallback for stdio or if OAuth already configured
    if isinstance(config, StdioMCPServerConfig):
        return False

    return not config.auth.oauth
