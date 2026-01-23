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
from dataclasses import dataclass, field, replace
import inspect
import time
from typing import TYPE_CHECKING, Any, Self, get_args, get_origin
from uuid import uuid4

import anyio
from pydantic import BaseModel, HttpUrl
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agentpool.agents import Agent
from agentpool.log import get_logger
from agentpool.resource_providers import ResourceChangeEvent
from agentpool.utils.signatures import filter_schema_params, get_params_matching_predicate


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from clawd_code_sdk.types import McpServerConfig
    from fastmcp import Context, FastMCP
    from fastmcp.tools.tool import ToolResult
    from pydantic_ai.messages import UserContent
    from uvicorn import Server

    from acp.schema.mcp import HttpMcpServer
    from agentpool.agents import AgentContext
    from agentpool.agents.base_agent import BaseAgent
    from agentpool.agents.prompt_injection import PromptInjectionManager
    from agentpool.tools.base import Tool
    from agentpool.ui.base import InputProvider
_ = ResourceChangeEvent  # Used at runtime in method signature


logger = get_logger(__name__)


def _is_agent_context_type(annotation: Any) -> bool:
    """Check if annotation is AgentContext."""
    if annotation is None or annotation is inspect.Parameter.empty:
        return False
    # Handle string annotations (forward references)
    if isinstance(annotation, str):
        base_name = annotation.split("[")[0].strip()
        return base_name == "AgentContext"
    # Check direct class match by name
    if isinstance(annotation, type) and annotation.__name__ == "AgentContext":
        return True
    # Check generic origin (e.g., AgentContext[SomeDeps])
    origin = get_origin(annotation)
    if origin is not None:
        if isinstance(origin, type) and origin.__name__ == "AgentContext":
            return True
        # Handle Union types (e.g., AgentContext | None)
        if origin is type(None) or str(origin) in ("typing.Union", "types.UnionType"):
            return any(_is_agent_context_type(arg) for arg in get_args(annotation))
    return False


def _get_context_param_names(fn: Callable[..., Any]) -> set[str]:
    """Get parameter names that are AgentContext (to be auto-injected)."""
    return get_params_matching_predicate(fn, lambda p: _is_agent_context_type(p.annotation))


def _is_run_context_type(annotation: Any) -> bool:
    """Check if annotation is pydantic-ai RunContext."""
    if annotation is None or annotation is inspect.Parameter.empty:
        return False
    # Handle string annotations (forward references)
    if isinstance(annotation, str):
        base_name = annotation.split("[")[0].strip()
        return base_name == "RunContext"
    # Check direct class match by name
    if isinstance(annotation, type) and annotation.__name__ == "RunContext":
        return True
    # Check generic origin (e.g., RunContext[SomeDeps])
    origin = get_origin(annotation)
    if origin is not None:
        if isinstance(origin, type) and origin.__name__ == "RunContext":
            return True
        # Handle Union types (e.g., RunContext | None)
        if origin is type(None) or str(origin) in ("typing.Union", "types.UnionType"):
            return any(_is_run_context_type(arg) for arg in get_args(annotation))
    return False


def _get_run_context_param_names(fn: Callable[..., Any]) -> set[str]:
    """Get parameter names that are RunContext (to be auto-injected)."""
    return get_params_matching_predicate(fn, lambda p: _is_run_context_type(p.annotation))


def _create_stub_run_context(
    ctx: AgentContext[Any],
    prompt: str | Sequence[UserContent] | None = None,
) -> RunContext[Any]:
    """Create a stub RunContext from AgentContext for MCP bridge tool invocations.

    This provides a best-effort RunContext for tools that require it when
    invoked via MCP bridge. Not all fields can be populated accurately since
    we're outside of pydantic-ai's normal execution flow.

    Args:
        ctx: The AgentContext available in the bridge
        prompt: The current prompt being processed

    Returns:
        A RunContext with available information populated
    """
    match ctx.agent:
        case Agent():
            model = ctx.agent._model or TestModel()
        # case ACPAgent() | ClaudeCodeAgent():
        #     try:
        #         model = infer_model(ctx.agent.model_name or "test")
        #     except Exception:
        #         model = TestModel()
        case _:
            model = TestModel()
    # Create a minimal usage object
    return RunContext(
        deps=ctx.data,
        model=model,
        usage=RunUsage(),
        prompt=prompt,
        messages=[],
        tool_name=ctx.tool_name,
        tool_call_id=ctx.tool_call_id,
    )


def _convert_to_tool_result(result: Any) -> ToolResult:
    """Convert a tool's return value to a FastMCP ToolResult.

    Handles different result types appropriately:
    - FastMCP ToolResult: Pass through unchanged
    - AgentPool ToolResult: Convert to FastMCP format
    - dict: Use as structured_content (enables programmatic access by clients)
    - Pydantic models: Serialize to dict for structured_content
    - Other types: Pass to ToolResult(content=...) which handles conversion internally
    """
    from fastmcp.tools.tool import ToolResult as FastMCPToolResult

    from agentpool.tools.base import ToolResult as AgentPoolToolResult

    # Already a FastMCP ToolResult - pass through
    if isinstance(result, FastMCPToolResult):
        return result
    # AgentPool ToolResult - convert to FastMCP format
    if isinstance(result, AgentPoolToolResult):
        return FastMCPToolResult(
            content=result.content,
            structured_content=result.structured_content,
            meta=result.metadata,
        )
    # Dict - use as structured_content (FastMCP auto-populates content as JSON)
    if isinstance(result, dict):
        return FastMCPToolResult(structured_content=result)
    # Pydantic model - serialize to dict for structured_content
    if isinstance(result, BaseModel):
        return FastMCPToolResult(structured_content=result.model_dump(mode="json"))
    # All other types (str, list, ContentBlock, Image, None, primitives, etc.)
    # ToolResult's internal _convert_to_content handles these correctly
    return FastMCPToolResult(content=result if result is not None else "")


def _append_injection_to_result(result: Any, injection: str) -> Any:
    """Append an injected message to a tool result.

    The injection is already wrapped in XML tags by the PromptInjectionManager.

    Handles different result types:
    - str: Append with newline separator
    - AgentPool ToolResult: Modify content field
    - dict: Add 'injected_context' key
    - Other: Wrap in dict with original and injection
    """
    from agentpool.tools.base import ToolResult as AgentPoolToolResult

    if isinstance(result, str):
        return f"{result}\n\n{injection}"

    if isinstance(result, AgentPoolToolResult):
        existing = result.content
        if existing is None:
            return replace(result, content=injection)
        if isinstance(existing, str):
            return replace(result, content=f"{existing}\n\n{injection}")
        # existing is a list
        return replace(result, content=[*existing, f"\n\n{injection}"])

    if isinstance(result, dict):
        return {**result, "injected_context": injection}

    # For other types, return tuple with original and injection
    return (result, {"injected_context": injection})


def _extract_tool_call_id(context: Context | None) -> str:
    """Extract Claude's original tool_call_id from request metadata.

    Claude Code passes the tool_use_id via the _meta field as
    'claudecode/toolUseId'. This allows us to maintain consistent
    tool_call_ids across ToolCallStartEvent and ToolCallCompleteEvent.

    Falls back to generating a UUID if not available.
    """
    if context and (request_ctx := context.request_context) and request_ctx.meta:
        # Access extra fields on the Meta object (extra="allow" in pydantic)
        meta_dict = request_ctx.meta.model_dump()
        claude_tool_id = meta_dict.get("claudecode/toolUseId")
        if isinstance(claude_tool_id, str):
            logger.debug("Extracted Claude tool_call_id", tool_call_id=claude_tool_id)
            return claude_tool_id
    return str(uuid4())  # Generate fallback UUID if no tool_call_id found in meta


@dataclass
class ToolManagerBridge:
    """Exposes a node's tools as an MCP server for ACP agents.

    This bridge allows external ACP agents to access our internal toolsets
    (SubagentTools, AgentManagementTools, etc.) via HTTP MCP transport.

    The node's existing context is used for tool invocations, providing
    pool access and proper configuration without reconstruction.

    Example:
        ```python
        bridge = ToolManagerBridge(node=agent)
        async with bridge.set_run_context(...):
            # Bridge is running, get MCP config for ACP agent
            mcp_config = bridge.get_mcp_server_config()
            # Pass to ACP agent...
        ```
    """

    node: BaseAgent[Any, Any]
    """The node whose tools to expose."""

    server_name: str | None = None
    """Name for the MCP server."""

    _current_deps: Any = field(default=None, init=False, repr=False)
    """Current dependencies for tool invocations (set by run_stream)."""

    _current_input_provider: InputProvider | None = field(default=None, init=False, repr=False)
    """Current input provider for tool invocations (set by run_stream)."""

    _current_prompt: str | Sequence[UserContent] | None = field(
        default=None, init=False, repr=False
    )
    """Current prompt for tool invocations (set by run_stream)."""

    _mcp: FastMCP | None = field(default=None, init=False, repr=False)
    """FastMCP server instance."""

    _server: Server | None = field(default=None, init=False, repr=False)
    """Uvicorn server instance."""

    _server_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    """Background task running the server."""

    _actual_port: int | None = field(default=None, init=False, repr=False)
    """Actual port the server is bound to."""

    injection_manager: PromptInjectionManager | None = field(default=None, repr=False)
    """Optional injection manager for mid-run prompt injection.

    When set, the bridge will consume pending injections after each tool call
    and append them to the tool result. This enables prompt injection for
    ACP agents (ACPAgent, ClaudeCodeAgent, CodexAgent) that use the bridge.
    """

    async def __aenter__(self) -> Self:
        """Start the MCP server."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the MCP server."""
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP MCP server in the background."""
        from fastmcp import FastMCP

        self._mcp = FastMCP(name=self.resolved_server_name)
        await self._register_tools()
        self._subscribe_to_tool_changes()
        await self._start_server()

    async def stop(self) -> None:
        """Stop the HTTP MCP server."""
        # Unsubscribe from tool changes
        self._unsubscribe_from_tool_changes()

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

    def _subscribe_to_tool_changes(self) -> None:
        """Subscribe to tool changes from all providers via signals."""
        for provider in self.node.tools.providers:
            provider.tools_changed.connect(self._on_tools_changed)

    def _unsubscribe_from_tool_changes(self) -> None:
        """Disconnect from tool change signals on all providers."""
        for provider in self.node.tools.providers:
            provider.tools_changed.disconnect(self._on_tools_changed)

    async def _on_tools_changed(self, event: ResourceChangeEvent) -> None:
        """Handle tool changes from a provider."""
        logger.info(
            "Tools changed in provider, refreshing MCP tools",
            provider=event.provider_name,
            provider_kind=event.provider_kind,
        )
        if self._mcp:
            await self._refresh_tools()

    async def _refresh_tools(self) -> None:
        """Refresh tools registered with the MCP server.

        Uses FastMCP's add_tool/remove_tool API which automatically sends
        ToolListChanged notifications when called within a request context.

        Note: FastMCP only sends notifications when inside a request context
        (ContextVar-based). Outside of requests, tools are updated but clients
        won't receive a push notification - they'll see changes on next list_tools.

        Future improvement: Access StreamableHTTPSessionManager._server_instances
        to broadcast ToolListChanged to all connected sessions regardless of context.
        """
        if not self._mcp:
            return

        # Get current and new tool sets
        # Support both old and new FastMCP API
        if hasattr(self._mcp, "_tool_manager"):
            # Old API (<=2.12.4): direct access to _tool_manager
            current_names = set(self._mcp._tool_manager._tools.keys())
        elif hasattr(self._mcp, "_local_provider"):
            # New API (git): tools stored in _local_provider._components
            # Keys are prefixed with 'tool:', e.g., 'tool:bash'
            current_names = {
                key.removeprefix("tool:")
                for key in self._mcp._local_provider._components  # pyright: ignore[reportAttributeAccessIssue]
                if key.startswith("tool:")
            }
        else:
            # Fallback: use async get_tools() method
            current_tools = await self._mcp.get_tools()
            current_names = {t.name for t in current_tools}  # type: ignore[attr-defined]
        new_tools = await self.node.tools.get_tools(state="enabled")
        new_names = {t.name for t in new_tools}

        # Remove tools that are no longer present
        for name in current_names - new_names:
            with suppress(Exception):
                self._mcp.remove_tool(name)

        # Add/update tools
        for tool in new_tools:
            if tool.name in current_names:
                # Remove and re-add to update
                with suppress(Exception):
                    self._mcp.remove_tool(tool.name)
            self._register_single_tool(tool)

    @property
    def port(self) -> int:
        """Get the actual port the server is running on."""
        if self._actual_port is None:
            raise RuntimeError("Server not started")
        return self._actual_port

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def resolved_server_name(self) -> str:
        """Get the server name."""
        return self.server_name or f"agentpool-{self.node.name}-tools"

    def get_mcp_server_config(self) -> HttpMcpServer:
        """Get ACP-compatible MCP server configuration.

        Returns config suitable for passing to ACP agent's NewSessionRequest.
        """
        from acp.schema import HttpMcpServer

        url = HttpUrl(self.url)
        return HttpMcpServer(name=self.resolved_server_name, url=url)

    def get_claude_mcp_server_config(self) -> dict[str, McpServerConfig]:
        """Get Claude Agent SDK-compatible MCP server configuration.

        Returns a dict suitable for passing to ClaudeAgentOptions.mcp_servers.
        Uses HTTP transport to preserve MCP _meta field which contains
        claudecode/toolUseId needed for proper tool_call_id correlation.

        Note: SDK transport (passing FastMCP instance directly) would be faster
        but the Claude Agent SDK's _handle_sdk_mcp_request drops the _meta field,
        breaking tool_call_id propagation for progress events.

        Returns:
            Dict mapping server name to McpHttpServerConfig

        Raises:
            RuntimeError: If bridge not started (no server running)
        """
        if self._actual_port is None:
            raise RuntimeError("Bridge not started - call start() first")

        # Use HTTP transport to preserve _meta field with claudecode/toolUseId
        # SDK transport drops _meta in Claude Agent SDK's query.py
        url = f"http://127.0.0.1:{self.port}/mcp"
        return {self.resolved_server_name: {"type": "http", "url": url}}

    def get_codex_mcp_server_config(self) -> tuple[str, Any]:
        """Get Codex app-server-compatible MCP server configuration.

        Returns:
            Tuple of (server name, HttpMcpServer config) suitable for adding
            to CodexClient mcp_servers dict.

        Raises:
            RuntimeError: If bridge not started (no server running)
        """
        from codex_adapter.codex_types import HttpMcpServer as CodexHttpMcpServer

        if self._actual_port is None:
            raise RuntimeError("Bridge not started - call start() first")

        url = f"http://127.0.0.1:{self.port}/mcp"
        return (self.resolved_server_name, CodexHttpMcpServer(url=url))

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
        from fastmcp.tools import Tool as FastMCPTool

        class _BridgeTool(FastMCPTool):
            """Custom FastMCP Tool that wraps a agentpool Tool.

            This allows us to use our own schema and invoke tools with AgentContext.
            """

            def __init__(self, tool: Tool, bridge: ToolManagerBridge) -> None:
                # Get input schema from our tool
                schema = tool.schema["function"]
                input_schema = schema.get("parameters", {"type": "object", "properties": {}})
                # Filter out context parameters - they're auto-injected by the bridge
                context_params = _get_context_param_names(tool.get_callable())
                run_context_params = _get_run_context_param_names(tool.get_callable())
                all_context_params = context_params | run_context_params
                filtered_schema = filter_schema_params(input_schema, all_context_params)
                desc = tool.description or "No description"
                super().__init__(name=tool.name, description=desc, parameters=filtered_schema)
                # Set these AFTER super().__init__() to avoid being overwritten
                self._tool = tool
                self._bridge = bridge

            async def run(self, arguments: dict[str, Any]) -> ToolResult:
                """Execute the wrapped tool with context bridging."""
                from fastmcp.server.dependencies import get_context

                from agentpool.agents.events import ToolResultMetadataEvent
                from agentpool.tools.base import ToolResult as AgentPoolToolResult

                # Validate arguments against tool's schema
                # try:
                #     param_model = self._tool.schema_obj.create_parameter_model()
                #     param_model.model_validate(arguments)
                # except pydantic.ValidationError as e:
                #     error_msg = (
                #         f"Tool '{self._tool.name}' called with invalid arguments. "
                #         f"Ensure arguments match the schema.\n\nValidation errors:\n{e}"
                #     )
                #     return ToolResult(content=[TextContent(type="text", text=error_msg)])

                # Get FastMCP context from context variable (not passed as parameter)
                try:
                    mcp_context: Context | None = get_context()
                except LookupError:
                    mcp_context = None
                # Try to get Claude's original tool_call_id from request metadata
                tc_id = _extract_tool_call_id(mcp_context)
                # Get deps and input_provider from bridge (set by run_stream on the agent)
                current_deps = self._bridge._current_deps
                current_input_provider = self._bridge._current_input_provider
                # Create context with tool-specific metadata from node's context.
                ctx = replace(
                    self._bridge.node.get_context(
                        data=current_deps,
                        input_provider=current_input_provider,
                    ),
                    tool_name=self._tool.name,
                    tool_call_id=tc_id,
                    tool_input=arguments,
                )
                # Invoke with context - copy arguments since invoke_tool_with_context
                # modifies kwargs in-place to inject context parameters
                result = await self._bridge.invoke_tool_with_context(
                    self._tool, ctx, arguments.copy()
                )
                # Emit metadata event for ClaudeCodeAgent to correlate
                # (works around Claude SDK stripping MCP _meta field)
                if isinstance(result, AgentPoolToolResult) and result.metadata:
                    logger.info("Emitting ToolResultMetadataEvent", tool_call_id=tc_id)
                    event = ToolResultMetadataEvent(tool_call_id=tc_id, metadata=result.metadata)
                    await ctx.events.emit_event(event)

                # Consume pending injection and append to result
                if self._bridge.injection_manager and (
                    injection := await self._bridge.injection_manager.consume()
                ):
                    result = _append_injection_to_result(result, injection)

                return _convert_to_tool_result(result)

        # Create a custom FastMCP Tool that wraps our tool
        bridge_tool = _BridgeTool(tool=tool, bridge=self)
        self._mcp.add_tool(bridge_tool)

    @asynccontextmanager
    async def set_run_context(
        self,
        deps: Any = None,
        input_provider: InputProvider | None = None,
        prompt: str | Sequence[UserContent] | None = None,
    ) -> AsyncIterator[Self]:
        """Context manager for setting run-scoped state.

        Ensures _current_deps, _current_input_provider, and _current_prompt are
        properly set for the duration of the run and cleaned up afterwards.

        Args:
            deps: Dependencies to set for tool invocations
            input_provider: Input provider to set for tool invocations
            prompt: Current prompt being processed
        """
        self._current_deps = deps
        self._current_input_provider = input_provider
        self._current_prompt = prompt
        try:
            yield self
        finally:
            self._current_deps = None
            self._current_input_provider = None
            self._current_prompt = None

    async def invoke_tool_with_context(
        self,
        tool: Tool,
        ctx: AgentContext[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Invoke a tool with proper context injection and hooks.

        Handles tools that expect AgentContext, RunContext, or neither.
        Runs pre/post tool hooks if configured on the node.
        """
        from agentpool.tasks import ToolSkippedError

        if hooks := self.node.hooks:
            pre_result = await hooks.run_pre_tool_hooks(
                agent_name=ctx.node_name,
                tool_name=tool.name,
                tool_input=kwargs,
                session_id=None,
            )
            if pre_result.get("decision") == "deny":
                reason = pre_result.get("reason", "Blocked by pre-tool hook")
                msg = f"Tool {tool.name} blocked: {reason}"
                raise ToolSkippedError(msg)
            # Apply modified input if provided
            if modified := pre_result.get("modified_input"):
                kwargs.update(modified)

        fn = tool.get_callable()
        # Inject AgentContext parameters
        context_param_names = _get_context_param_names(fn)
        for param_name in context_param_names:
            if param_name not in kwargs:
                kwargs[param_name] = ctx
        # Inject RunContext parameters (as stub since we're outside pydantic-ai)
        run_context_param_names = _get_run_context_param_names(fn)
        if run_context_param_names:
            stub_run_ctx = _create_stub_run_context(ctx, prompt=self._current_prompt)
            for param_name in run_context_param_names:
                if param_name not in kwargs:
                    kwargs[param_name] = stub_run_ctx

        start_time = time.perf_counter()
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        duration_ms = (time.perf_counter() - start_time) * 1000
        if hooks:
            await hooks.run_post_tool_hooks(
                agent_name=ctx.node_name,
                tool_name=tool.name,
                tool_input=kwargs,
                tool_output=result,
                duration_ms=duration_ms,
                session_id=None,
            )

        return result

    async def _start_server(self) -> None:
        """Start the uvicorn server in the background."""
        import socket

        import uvicorn

        if not self._mcp:
            msg = "MCP server not initialized"
            raise RuntimeError(msg)

        # Auto-select an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._actual_port = port
        app = self._mcp.http_app(transport="http")  # Create the ASGI app
        cfg = uvicorn.Config(
            app=app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio"
        )
        self._server = uvicorn.Server(cfg)
        # Start server in background task
        name = f"mcp-bridge-{self.resolved_server_name}"
        self._server_task = asyncio.create_task(self._server.serve(), name=name)
        await anyio.sleep(0.1)  # Wait briefly for server to start
        msg = "ToolManagerBridge started"
        logger.info(msg, url=self.url)
