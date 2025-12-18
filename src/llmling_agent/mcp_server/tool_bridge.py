"""MCP server bridge for exposing ToolManager tools to ACP agents.

This module provides a bridge that exposes a ToolManager's tools as an MCP server
using HTTP transport. This allows ACP agents (external agents like Claude Code,
Gemini CLI, etc.) to use our internal toolsets like SubagentTools,
AgentManagementTools, etc.

The bridge runs in-process on the same event loop, providing direct access to
the pool and avoiding IPC serialization overhead.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPTool
from pydantic import HttpUrl

from llmling_agent.log import get_logger
from llmling_agent.utils.signatures import (
    filter_schema_params,
    get_params_matching_predicate,
    is_context_type,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fastmcp import Context
    from fastmcp.tools.tool import ToolResult
    from uvicorn import Server

    from acp.schema.mcp import HttpMcpServer, SseMcpServer
    from llmling_agent.agents.base_agent import BaseAgent
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


def _get_context_param_names(fn: Callable[..., Any]) -> set[str]:
    """Get names of parameters that are context types (to be auto-injected)."""
    return get_params_matching_predicate(fn, lambda p: is_context_type(p.annotation))


@dataclass
class BridgeConfig:
    """Configuration for the ToolManager MCP bridge."""

    host: str = "127.0.0.1"
    """Host to bind the HTTP server to."""

    port: int = 0
    """Port to bind to (0 = auto-select available port)."""

    transport: str = "sse"
    """Transport protocol: 'sse' or 'streamable-http'."""

    server_name: str = "llmling-toolmanager"
    """Name for the MCP server."""


@dataclass
class ToolManagerBridge:
    """Exposes a node's tools as an MCP server for ACP agents.

    This bridge allows external ACP agents to access our internal toolsets
    (SubagentTools, AgentManagementTools, etc.) via HTTP MCP transport.

    The node's existing context is used for tool invocations, providing
    pool access and proper configuration without reconstruction.

    Example:
        ```python
        async with AgentPool() as pool:
            agent = pool.agents["my_agent"]
            bridge = ToolManagerBridge(node=agent, config=BridgeConfig(port=8765))
            async with bridge:
                # Bridge is running, get MCP config for ACP agent
                mcp_config = bridge.get_mcp_server_config()
                # Pass to ACP agent...
        ```
    """

    node: BaseAgent[Any, Any]
    """The node whose tools to expose."""

    config: BridgeConfig = field(default_factory=BridgeConfig)
    """Bridge configuration."""

    _mcp: FastMCP | None = field(default=None, init=False, repr=False)
    """FastMCP server instance."""

    _server: Server | None = field(default=None, init=False, repr=False)
    """Uvicorn server instance."""

    _server_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    """Background task running the server."""

    _actual_port: int | None = field(default=None, init=False, repr=False)
    """Actual port the server is bound to."""

    async def __aenter__(self) -> Self:
        """Start the MCP server."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the MCP server."""
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP MCP server in the background."""
        self._mcp = FastMCP(name=self.config.server_name)
        await self._register_tools()
        await self._start_server()

    async def stop(self) -> None:
        """Stop the HTTP MCP server."""
        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task
            self._server = None
            self._server_task = None
        self._mcp = None
        self._actual_port = None
        logger.info("ToolManagerBridge stopped")

    @property
    def port(self) -> int:
        """Get the actual port the server is running on."""
        if self._actual_port is None:
            msg = "Server not started"
            raise RuntimeError(msg)
        return self._actual_port

    @property
    def url(self) -> str:
        """Get the server URL."""
        path = "/sse" if self.config.transport == "sse" else "/mcp"
        return f"http://{self.config.host}:{self.port}{path}"

    def get_mcp_server_config(self) -> HttpMcpServer | SseMcpServer:
        """Get ACP-compatible MCP server configuration.

        Returns config suitable for passing to ACP agent's NewSessionRequest.
        """
        from acp.schema.mcp import HttpMcpServer, SseMcpServer

        url = HttpUrl(self.url)
        if self.config.transport == "sse":
            return SseMcpServer(name=self.config.server_name, url=url, headers=[])
        return HttpMcpServer(name=self.config.server_name, url=url, headers=[])

    async def _register_tools(self) -> None:
        """Register all node tools with the FastMCP server."""
        if not self._mcp:
            return

        tools = await self.node.tools.get_tools(state="enabled")
        for tool in tools:
            self._register_single_tool(tool)
        logger.info("Registered tools with MCP bridge", tools=[t.name for t in tools])

    def _register_single_tool(self, tool: Tool) -> None:
        """Register a single tool with the FastMCP server."""
        if not self._mcp:
            return
        # Create a custom FastMCP Tool that wraps our tool
        bridge_tool = _BridgeTool(tool=tool, bridge=self)
        self._mcp.add_tool(bridge_tool)

    async def invoke_tool_with_context(
        self,
        tool: Tool,
        ctx: NodeContext[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Invoke a tool with proper context injection.

        Handles tools that expect AgentContext, RunContext, or neither.
        """
        fn = tool.callable

        # Find context parameters and inject them
        context_param_names = _get_context_param_names(fn)
        for param_name in context_param_names:
            # Don't override if caller somehow provided it (shouldn't happen)
            if param_name not in kwargs:
                kwargs[param_name] = ctx

        # Execute the tool
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result

        return result

    async def _start_server(self) -> None:
        """Start the uvicorn server in the background."""
        import socket

        import uvicorn

        if not self._mcp:
            msg = "MCP server not initialized"
            raise RuntimeError(msg)

        # Determine actual port (auto-select if 0)
        port = self.config.port
        if port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.config.host, 0))
                port = s.getsockname()[1]
        self._actual_port = port
        # Create the ASGI app
        app = self._mcp.http_app(transport=self.config.transport)  # type: ignore[arg-type]
        # Configure uvicorn
        cfg = uvicorn.Config(app=app, host=self.config.host, port=port, log_level="warning")
        self._server = uvicorn.Server(cfg)
        # Start server in background task
        name = f"mcp-bridge-{self.config.server_name}"
        self._server_task = asyncio.create_task(self._server.serve(), name=name)
        await asyncio.sleep(0.1)  # Wait briefly for server to start
        msg = "ToolManagerBridge started"
        logger.info(msg, url=self.url, transport=self.config.transport)


class _BridgeTool(FastMCPTool):
    """Custom FastMCP Tool that wraps a llmling-agent Tool.

    This allows us to use our own schema and invoke tools with AgentContext.
    """

    def __init__(self, tool: Tool, bridge: ToolManagerBridge) -> None:
        # Get input schema from our tool
        schema = tool.schema["function"]
        input_schema = schema.get("parameters", {"type": "object", "properties": {}})

        # Filter out context parameters - they're auto-injected by the bridge
        context_params = _get_context_param_names(tool.callable)
        filtered_schema = filter_schema_params(input_schema, context_params)

        desc = tool.description or "No description"
        super().__init__(name=tool.name, description=desc, parameters=filtered_schema)
        # Set these AFTER super().__init__() to avoid being overwritten
        self._tool = tool
        self._bridge = bridge

    async def run(self, arguments: dict[str, Any], context: Context | None = None) -> ToolResult:
        """Execute the wrapped tool with context bridging."""
        from fastmcp.tools.tool import ToolResult

        from llmling_agent.agents.context import AgentContext

        tool_call_id = str(uuid4())
        node_ctx = self._bridge.node.context
        # Create AgentContext with tool-specific metadata
        ctx = AgentContext(
            node=node_ctx.node,
            pool=node_ctx.pool,
            config=node_ctx.config,  # type: ignore[arg-type]
            definition=node_ctx.definition,
            input_provider=node_ctx.input_provider,
            data=node_ctx.data,
            tool_name=self._tool.name,
            tool_call_id=tool_call_id,
            tool_input=arguments,
        )

        # Invoke with context
        result = await self._bridge.invoke_tool_with_context(self._tool, ctx, arguments)
        # Convert result to ToolResult
        if isinstance(result, str):
            return ToolResult(content=result)
        return ToolResult(content=str(result))


@asynccontextmanager
async def create_tool_bridge(
    node: BaseAgent[Any, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    transport: str = "sse",
) -> AsyncIterator[ToolManagerBridge]:
    """Create and start a ToolManagerBridge as a context manager.

    Args:
        node: The node whose tools to expose
        host: Host to bind to
        port: Port to bind to (0 = auto-select)
        transport: Transport protocol ('sse' or 'streamable-http')

    Yields:
        Running ToolManagerBridge instance
    """
    config = BridgeConfig(host=host, port=port, transport=transport)
    bridge = ToolManagerBridge(node=node, config=config)
    async with bridge:
        yield bridge


class ToolBridgeRegistry:
    """Registry for managing multiple tool bridges.

    Useful when multiple ACP agents need access to different toolsets.
    """

    def __init__(self) -> None:
        self._bridges: dict[str, ToolManagerBridge] = {}
        self._port_counter = 18000  # Start port range for auto-allocation

    async def create_bridge(
        self,
        name: str,
        node: BaseAgent[Any, Any],
    ) -> ToolManagerBridge:
        """Create and register a new bridge.

        Args:
            name: Unique name for this bridge
            node: The node whose tools to expose

        Returns:
            Started ToolManagerBridge
        """
        if name in self._bridges:
            msg = f"Bridge {name!r} already exists"
            raise ValueError(msg)

        config = BridgeConfig(port=self._port_counter, server_name=f"llmling-{name}")
        self._port_counter += 1

        bridge = ToolManagerBridge(node=node, config=config)
        await bridge.start()
        self._bridges[name] = bridge
        return bridge

    async def get_bridge(self, name: str) -> ToolManagerBridge:
        """Get a bridge by name."""
        if name not in self._bridges:
            msg = f"Bridge {name!r} not found"
            raise KeyError(msg)
        return self._bridges[name]

    async def remove_bridge(self, name: str) -> None:
        """Stop and remove a bridge."""
        if name in self._bridges:
            await self._bridges[name].stop()
            del self._bridges[name]

    async def close_all(self) -> None:
        """Stop all bridges."""
        for bridge in list(self._bridges.values()):
            await bridge.stop()
        self._bridges.clear()

    def get_all_mcp_configs(self) -> list[HttpMcpServer | SseMcpServer]:
        """Get MCP server configs for all active bridges."""
        return [bridge.get_mcp_server_config() for bridge in self._bridges.values()]
