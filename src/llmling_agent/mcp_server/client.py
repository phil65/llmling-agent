"""FastMCP-based client implementation for LLMling agent.

This module provides a client for communicating with MCP servers using FastMCP.
It includes support for contextual progress handlers that extend FastMCP's
standard progress callbacks with tool execution context (tool name, call ID, and input).

The key innovation is the signature injection system that allows MCP tools to work
seamlessly with PydanticAI's RunContext while providing rich progress information.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Self, assert_never

from pydantic_ai import RunContext, ToolReturn
from schemez import FunctionSchema

from llmling_agent.agent.context import AgentContext
from llmling_agent.log import get_logger
from llmling_agent.mcp_server.constants import MCP_TO_LOGGING
from llmling_agent.mcp_server.helpers import (
    extract_text_content,
    mcp_tool_to_fn_schema,
)
from llmling_agent.mcp_server.message_handler import MCPMessageHandler
from llmling_agent.tools.base import Tool
from llmling_agent.utils.signatures import create_modified_signature
from llmling_agent_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from fastmcp.client import ClientTransport
    from fastmcp.client.elicitation import ElicitationHandler
    from fastmcp.client.logging import LogMessage
    from fastmcp.client.messages import MessageHandler, MessageHandlerT
    from fastmcp.client.sampling import ClientSamplingHandler
    from mcp.types import (
        BlobResourceContents,
        ContentBlock,
        GetPromptResult,
        Prompt as MCPPrompt,
        Resource as MCPResource,
        TextResourceContents,
        Tool as MCPTool,
    )
    from pydantic_ai import BinaryContent

    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


class MCPClient:
    """FastMCP-based client for communicating with MCP servers."""

    def __init__(
        self,
        config: MCPServerConfig,
        elicitation_callback: ElicitationHandler | None = None,
        sampling_callback: ClientSamplingHandler | None = None,
        message_handler: MessageHandlerT | MessageHandler | None = None,
        accessible_roots: list[str] | None = None,
        tool_change_callback: Callable[[], Awaitable[None]] | None = None,
        prompt_change_callback: Callable[[], Awaitable[None]] | None = None,
        resource_change_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._elicitation_callback = elicitation_callback
        self.config = config
        self._sampling_callback = sampling_callback
        # Store message handler or mark for lazy creation
        self._message_handler = message_handler
        self._accessible_roots = accessible_roots or []
        self._tool_change_callback = tool_change_callback
        self._prompt_change_callback = prompt_change_callback
        self._resource_change_callback = resource_change_callback
        self._client = self._get_client(self.config)

    @property
    def connected(self) -> bool:
        """Check if client is connected by examining session state."""
        return self._client.is_connected()

    async def __aenter__(self) -> Self:
        """Enter context manager."""
        try:
            # First attempt with configured auth
            await self._client.__aenter__()

        except Exception as first_error:
            # OAuth fallback for HTTP/SSE if not already using OAuth
            if (
                not isinstance(self.config, StdioMCPServerConfig)
                and not self.config.auth.oauth
            ):
                try:
                    with contextlib.suppress(Exception):
                        await self._client.__aexit__(None, None, None)
                    self._client = self._get_client(self.config, force_oauth=True)
                    await self._client.__aenter__()
                    logger.info("Connected with OAuth fallback")
                except Exception:  # noqa: BLE001
                    raise first_error from None
            else:
                raise

        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit context manager and cleanup."""
        try:
            await self._client.__aexit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error during FastMCP client cleanup", error=e)

    def get_resource_fs(self):
        """Get a filesystem for accessing MCP resources."""
        from upathtools.filesystems.mcp_fs import MCPFileSystem

        return MCPFileSystem(client=self._client)

    async def _log_handler(self, message: LogMessage) -> None:
        """Handle server log messages."""
        level = MCP_TO_LOGGING.get(message.level, logging.INFO)
        logger.log(level, "MCP Server: ", data=message.data)

    def _get_client(self, config: MCPServerConfig, force_oauth: bool = False):
        """Create FastMCP client based on config."""
        import fastmcp
        from fastmcp.client import SSETransport, StreamableHttpTransport
        from fastmcp.client.transports import StdioTransport

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
            case _ as unreachable:
                assert_never(unreachable)

        # Create message handler if needed
        msg_handler = self._message_handler or MCPMessageHandler(
            self,
            self._tool_change_callback,
            self._prompt_change_callback,
            self._resource_change_callback,
        )
        return fastmcp.Client(
            transport,
            log_handler=self._log_handler,
            roots=self._accessible_roots,
            timeout=config.timeout,
            elicitation_handler=self._elicitation_callback,
            sampling_handler=self._sampling_callback,
            message_handler=msg_handler,
            auth="oauth" if (force_oauth or oauth) else None,
        )

    async def list_tools(self) -> list[MCPTool]:
        """Get available tools directly from the server."""
        if not self.connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            tools = await self._client.list_tools()
            logger.debug("Listed tools from MCP server", num_tools=len(tools))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to list tools", error=e)
            return []
        else:
            return tools

    async def list_prompts(self) -> list[MCPPrompt]:
        """Get available prompts from the server."""
        if not self.connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.list_prompts()
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to list prompts", error=e)
            return []

    async def list_resources(self) -> list[MCPResource]:
        """Get available resources from the server."""
        if not self.connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.list_resources()
        except Exception as e:
            msg = f"Failed to list resources: {e}"
            raise RuntimeError(msg) from e

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> GetPromptResult:
        """Get a specific prompt's content."""
        if not self.connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.get_prompt_mcp(name, arguments)
        except Exception as e:
            msg = f"Failed to get prompt {name!r}: {e}"
            raise RuntimeError(msg) from e

    def convert_tool(self, tool: MCPTool) -> Tool:
        """Create a properly typed callable from MCP tool schema."""

        async def tool_callable(
            ctx: RunContext, agent_ctx: AgentContext[Any], **kwargs: Any
        ) -> str | Any | ToolReturn:
            """Dynamically generated MCP tool wrapper."""
            # Filter out None values for optional params
            schema_props = tool.inputSchema.get("properties", {})
            required_props = set(tool.inputSchema.get("required", []))
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in required_props or (k in schema_props and v is not None)
            }
            return await self.call_tool(tool.name, ctx, filtered_kwargs, agent_ctx)

        # Set proper signature and annotations with both RunContext and AgentContext
        schema = mcp_tool_to_fn_schema(tool)
        fn_schema = FunctionSchema.from_dict(schema)
        sig = fn_schema.to_python_signature()

        tool_callable.__signature__ = create_modified_signature(  # type: ignore[attr-defined]
            sig, inject={"ctx": RunContext, "agent_ctx": AgentContext}
        )
        annotations = fn_schema.get_annotations()
        annotations["ctx"] = RunContext
        annotations["agent_ctx"] = AgentContext
        # Update return annotation to support multiple types
        annotations["return"] = str | Any | ToolReturn  # type: ignore
        tool_callable.__annotations__ = annotations
        tool_callable.__name__ = tool.name
        tool_callable.__doc__ = tool.description or "No description provided."
        return Tool.from_callable(tool_callable, source="mcp")

    async def call_tool(
        self,
        name: str,
        run_context: RunContext,
        arguments: dict[str, Any] | None = None,
        agent_ctx: AgentContext[Any] | None = None,
    ) -> ToolReturn | str | Any:
        """Call an MCP tool with full PydanticAI return type support."""
        if not self.connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        # Create progress handler that bridges to AgentContext if available
        progress_handler = None
        if agent_ctx:

            async def fastmcp_progress_handler(
                progress: float,
                total: float | None,
                message: str | None,
            ) -> None:
                await agent_ctx.report_progress(progress, total, message or "")

            progress_handler = fastmcp_progress_handler

        try:
            result = await self._client.call_tool(
                name, arguments, progress_handler=progress_handler
            )
            content = await self._convert_mcp_content(result.content)
            # Decision logic for return type
            match (result.data is not None, bool(content)):
                case (True, True):  # Both structured data and rich content -> ToolReturn
                    return ToolReturn(return_value=result.data, content=content)
                case (True, False):  # Only structured data -> return directly
                    return result.data
                case (False, True):  # Only content -> ToolReturn with content
                    msg = "Tool executed successfully"
                    return ToolReturn(return_value=msg, content=content)
                case (False, False):  # Fallback to text extraction
                    return extract_text_content(result.content)
                case _:  # Handle unexpected cases
                    msg = f"Unexpected MCP content: {result.content}"
                    raise ValueError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"MCP tool call failed: {e}"
            raise RuntimeError(msg) from e

    async def _convert_mcp_content(
        self,
        mcp_content: Sequence[ContentBlock | TextResourceContents | BlobResourceContents],
    ) -> list[str | BinaryContent]:
        """Convert MCP content blocks to PydanticAI content types."""
        from llmling_agent.mcp_server.conversions import convert_mcp_content

        return await convert_mcp_content(mcp_content)


if __name__ == "__main__":
    import asyncio

    path = "/home/phil65/dev/oss/llmling-agent/tests/mcp/server.py"
    # path = Path(__file__).parent / "test_mcp_server.py"
    config = StdioMCPServerConfig(
        command="uv",
        args=["run", str(path)],
    )

    async def main() -> None:
        async with MCPClient(config=config) as mcp_client:
            # Create MCP filesystem
            fs = mcp_client.get_resource_fs()
            print(await fs._ls(""))

    asyncio.run(main())
