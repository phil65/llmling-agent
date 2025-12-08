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


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastmcp import Context
    from fastmcp.tools.tool import ToolResult
    from uvicorn import Server

    from acp.schema.mcp import HttpMcpServer, SseMcpServer
    from llmling_agent import AgentPool
    from llmling_agent.agent.context import AgentContext
    from llmling_agent.models.agents import AgentConfig
    from llmling_agent.models.manifest import AgentsManifest
    from llmling_agent.tools.base import Tool
    from llmling_agent.tools.manager import ToolManager
    from llmling_agent.ui.base import InputProvider


logger = get_logger(__name__)


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
    """Exposes a ToolManager's tools as an MCP server for ACP agents.

    This bridge allows external ACP agents to access our internal toolsets
    (SubagentTools, AgentManagementTools, etc.) via HTTP MCP transport.

    The bridge creates synthetic AgentContext instances for tool invocations,
    allowing tools that need pool access to work correctly.

    Example:
        ```python
        async with AgentPool() as pool:
            agent = pool.agents["my_agent"]
            bridge = ToolManagerBridge(
                tool_manager=agent.tools,
                pool=pool,
                config=BridgeConfig(port=8765),
            )
            async with bridge:
                # Bridge is running, get MCP config for ACP agent
                mcp_config = bridge.get_mcp_server_config()
                # Pass to ACP agent...
        ```
    """

    tool_manager: ToolManager
    """The ToolManager whose tools to expose."""

    pool: AgentPool[Any]
    """Agent pool for context creation."""

    config: BridgeConfig = field(default_factory=BridgeConfig)
    """Bridge configuration."""

    owner_agent_name: str | None = None
    """Name of the agent that owns this bridge (for context creation)."""

    input_provider: InputProvider | None = None
    """Optional input provider for tool confirmations."""

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

    def _create_proxy_context(
        self,
        tool_name: str,
        tool_call_id: str,
        tool_input: dict[str, Any],
        mcp_ctx: Context,
    ) -> AgentContext[None]:
        """Create a synthetic AgentContext for MCP tool invocation.

        This context provides pool access for delegation tools while
        bridging progress reporting to MCP.
        """
        from llmling_agent.agent.context import AgentContext
        from llmling_agent.models.agents import AgentConfig
        from llmling_agent.models.manifest import AgentsManifest

        # Use owner agent's config or create minimal one
        agent_config: AgentConfig
        definition: AgentsManifest
        if self.owner_agent_name and self.owner_agent_name in self.pool.agents:
            agent = self.pool.agents[self.owner_agent_name]
            agent_config = agent.context.config
            definition = agent.context.definition
        else:
            # Create minimal config for bridge-only usage
            agent_config = AgentConfig(name="bridge")
            definition = AgentsManifest()

        return AgentContext(
            node_name=self.owner_agent_name or "bridge",
            pool=self.pool,
            config=agent_config,
            definition=definition,
            input_provider=self.input_provider,
            data=None,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
        )

    async def _register_tools(self) -> None:
        """Register all ToolManager tools with the FastMCP server."""
        if not self._mcp:
            return

        tools = await self.tool_manager.get_tools(state="enabled")
        for tool in tools:
            self._register_single_tool(tool)
        msg = "Registered tools with MCP bridge"
        logger.info(msg, tool_count=len(tools), tools=[t.name for t in tools])

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
        agent_ctx: AgentContext[None],
        kwargs: dict[str, Any],
    ) -> Any:
        """Invoke a tool with proper context injection.

        Handles tools that expect AgentContext, RunContext, or neither.
        """
        from llmling_agent.agent.context import AgentContext as AgentContextType

        fn = tool.callable
        sig = inspect.signature(fn)

        # Check what context types the tool expects
        needs_agent_ctx = any(
            param.annotation is AgentContextType
            or (
                hasattr(param.annotation, "__origin__")
                and param.annotation.__origin__ is AgentContextType
            )
            or str(param.annotation).startswith("AgentContext")
            for param in sig.parameters.values()
        )

        # Find parameter name for AgentContext if needed
        agent_ctx_param: str | None = None
        if needs_agent_ctx:
            for name, param in sig.parameters.items():
                annotation = param.annotation
                if annotation is AgentContextType or str(annotation).startswith("AgentContext"):
                    agent_ctx_param = name
                    break

        # Invoke with appropriate context injection
        if agent_ctx_param:
            kwargs[agent_ctx_param] = agent_ctx

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
        desc = tool.description or "No description"
        super().__init__(name=tool.name, description=desc, parameters=input_schema)
        # Set these AFTER super().__init__() to avoid being overwritten
        self._tool = tool
        self._bridge = bridge

    async def run(self, arguments: dict[str, Any], context: Context | None = None) -> ToolResult:
        """Execute the wrapped tool with context bridging."""
        from fastmcp.tools.tool import ToolResult

        tool_call_id = str(uuid4())
        # Create proxy context with pool access
        agent_ctx = self._bridge._create_proxy_context(
            tool_name=self._tool.name,
            tool_call_id=tool_call_id,
            tool_input=arguments,
            mcp_ctx=context,
        )

        # Invoke with context
        result = await self._bridge.invoke_tool_with_context(self._tool, agent_ctx, arguments)
        # Convert result to ToolResult
        if isinstance(result, str):
            return ToolResult(content=result)
        return ToolResult(content=str(result))


@asynccontextmanager
async def create_tool_bridge(
    tool_manager: ToolManager,
    pool: AgentPool[Any],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    transport: str = "sse",
    owner_agent_name: str | None = None,
    input_provider: InputProvider | None = None,
) -> AsyncIterator[ToolManagerBridge]:
    """Create and start a ToolManagerBridge as a context manager.

    Args:
        tool_manager: ToolManager whose tools to expose
        pool: Agent pool for context creation
        host: Host to bind to
        port: Port to bind to (0 = auto-select)
        transport: Transport protocol ('sse' or 'streamable-http')
        owner_agent_name: Name of owning agent for context
        input_provider: Optional input provider

    Yields:
        Running ToolManagerBridge instance
    """
    config = BridgeConfig(host=host, port=port, transport=transport)
    bridge = ToolManagerBridge(
        tool_manager=tool_manager,
        pool=pool,
        config=config,
        owner_agent_name=owner_agent_name,
        input_provider=input_provider,
    )
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
        tool_manager: ToolManager,
        pool: AgentPool[Any],
        *,
        owner_agent_name: str | None = None,
        input_provider: InputProvider | None = None,
    ) -> ToolManagerBridge:
        """Create and register a new bridge.

        Args:
            name: Unique name for this bridge
            tool_manager: ToolManager to expose
            pool: Agent pool for context
            owner_agent_name: Optional owner agent name
            input_provider: Optional input provider

        Returns:
            Started ToolManagerBridge
        """
        if name in self._bridges:
            msg = f"Bridge {name!r} already exists"
            raise ValueError(msg)

        config = BridgeConfig(port=self._port_counter, server_name=f"llmling-{name}")
        self._port_counter += 1

        bridge = ToolManagerBridge(
            tool_manager=tool_manager,
            pool=pool,
            config=config,
            owner_agent_name=owner_agent_name,
            input_provider=input_provider,
        )
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
