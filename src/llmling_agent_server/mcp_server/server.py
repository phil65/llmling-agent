"""MCP protocol server implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from key_value.aio.stores.disk import DiskStore
from mcp import CreateMessageResult
from mcp.types import TextContent
import platformdirs

import llmling_agent
from llmling_agent.log import get_logger
from llmling_agent_server import BaseServer
from llmling_agent_server.mcp_server.handlers import register_handlers


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager
    from typing import Any

    from fastmcp import SamplingMessage, ServerSession
    import mcp
    from mcp.server.lowlevel.server import LifespanResultT
    from mcp.shared.context import LifespanContextT, RequestContext
    from mcp.types import CreateMessageRequestParams as SamplingParams

    from llmling_agent import AgentPool
    from llmling_agent_config.pool_server import MCPPoolServerConfig

    LifespanHandler = Callable[
        [FastMCP[LifespanResultT]],
        AbstractAsyncContextManager[LifespanResultT],
    ]

logger = get_logger(__name__)


llmling_dir = platformdirs.user_config_dir("llmling-agent")

store = DiskStore(directory=llmling_dir)
middleware = ResponseCachingMiddleware(cache_storage=store)


class MCPServer(BaseServer):
    """MCP protocol server implementation."""

    def __init__(
        self,
        pool: AgentPool[Any],
        config: MCPPoolServerConfig,
        lifespan: (LifespanHandler | None) = None,
        instructions: str | None = None,
        name: str = "llmling-server",
        raise_exceptions: bool = False,
    ) -> None:
        """Initialize server with agent pool.

        Args:
            pool: AgentPool to expose through MCP
            config: Server configuration
            name: Server name for MCP protocol
            lifespan: Lifespan context manager
            instructions: Instructions for Server usage
            raise_exceptions: Whether to raise exceptions during server start
        """
        from llmling_agent.resource_providers.pool import PoolResourceProvider

        super().__init__(pool, name=name, raise_exceptions=raise_exceptions)
        self.provider = PoolResourceProvider(pool, zed_mode=config.zed_mode)
        self.config = config

        # Handle Zed mode if enabled
        if config.zed_mode:
            pass
            # TODO: adapt zed wrapper to work with ResourceProvider
            # prepare_runtime_for_zed(runtime)

        self._subscriptions: defaultdict[str, set[mcp.ServerSession]] = defaultdict(set)

        self.fastmcp = FastMCP(
            instructions=instructions,
            lifespan=lifespan,
            version=llmling_agent.__version__,
            middleware=[middleware],
            # sampling_handler=self._sampling_handler,
        )
        self.server = self.fastmcp._mcp_server
        register_handlers(self)

    async def _sampling_handler(
        self,
        messages: list[SamplingMessage],
        params: SamplingParams,
        context: RequestContext[ServerSession, LifespanContextT],
    ) -> CreateMessageResult:
        # This is a fallback handler in case the client does not support sampling.
        return CreateMessageResult(
            role="user",
            content=TextContent(type="text", text="test"),
            model="test",
        )

    async def _start_async(self) -> None:
        """Start the server (blocking async - runs until stopped)."""
        if self.config.transport == "stdio":
            await self.fastmcp.run_async(transport=self.config.transport)
        else:
            await self.fastmcp.run_async(
                transport=self.config.transport,
                host=self.config.host,
                port=self.config.port,
            )

    @property
    def current_session(self) -> mcp.ServerSession:
        """Get current session from request context."""
        try:
            return self.server.request_context.session
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
        related_request_id: str | None = None,
    ) -> None:
        """Report progress for the current operation."""
        progress_token = (
            self.server.request_context.meta.progressToken
            if self.server.request_context.meta
            else None
        )

        if progress_token is None:
            return

        await self.server.request_context.session.send_progress_notification(
            progress_token=progress_token,
            progress=progress,
            total=total,
            message=message,
            related_request_id=related_request_id,
        )

    @property
    def client_info(self) -> mcp.Implementation | None:
        """Get client info from current session."""
        session = self.current_session
        if not session.client_params:
            return None
        return session.client_params.clientInfo

    async def notify_tool_list_changed(self) -> None:
        """Notify clients about tool list changes."""
        try:
            self.task_manager.create_task(self.current_session.send_tool_list_changed())
        except RuntimeError:
            self.log.debug("No active session for notification")
        except Exception:
            self.log.exception("Failed to send tool list change notification")

    async def notify_prompt_list_changed(self) -> None:
        """Notify clients about prompt list changes."""
        try:
            self.task_manager.create_task(self.current_session.send_prompt_list_changed())
        except RuntimeError:
            self.log.debug("No active session for notification")
        except Exception:
            self.log.exception("Failed to send prompt list change notification")
