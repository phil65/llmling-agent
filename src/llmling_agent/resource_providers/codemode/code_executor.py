"""Code execution provider with secure tool isolation via FastAPI server."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence
    import socket
    from types import TracebackType

    from exxec.base import ExecutionEnvironment
    from exxec.configs import ExecutionEnvironmentConfig
    from exxec.models import ServerInfo
    from fastapi import FastAPI
    from schemez import ToolsetCodeGenerator
    import uvicorn

    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


@dataclass
class RemoteCodeExecutor:
    """Provides secure code execution with tool access via FastAPI server.

    Architecture:
    - FastAPI server runs in HOST environment with tool routes
    - User code runs in SANDBOX environment (Docker, E2B, etc.)
    - Sandbox makes HTTP calls to server for tool execution
    """

    toolset_generator: ToolsetCodeGenerator
    """Code generator for tools."""

    execution_env: ExecutionEnvironment
    """Execution environment for running code."""

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[Tool],
        env_config: ExecutionEnvironmentConfig,
        server_host: str = "localhost",
        server_port: int = 8000,
        include_docstrings: bool = True,
    ) -> RemoteCodeExecutor:
        """Create provider from tools and environment configuration.

        Args:
            tools: Tools to make available for code execution
            env_config: Execution environment configuration
            server_host: Host for FastAPI server
            server_port: Port for FastAPI server
            include_docstrings: Include function docstrings in documentation

        Returns:
            RemoteCodeExecutor instance
        """
        from llmling_agent.resource_providers.codemode.helpers import tools_to_codegen

        toolset_gen = tools_to_codegen(tools, include_docstrings)
        server_handler = ToolServerLifecycleHandler(toolset_gen, server_host, server_port)
        execution_env = env_config.get_provider(server_handler)
        return cls(toolset_gen, execution_env)

    def get_tool_description(self) -> str:
        """Get comprehensive description of available tools."""
        return self.toolset_generator.generate_tool_description()

    async def execute_code(self, code: str) -> Any:
        """Execute code with tools available via HTTP API.

        Args:
            code: Python code to execute

        Returns:
            Execution result from the environment
        """
        return await self.execution_env.execute(code)

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.execution_env.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        return await self.execution_env.__aexit__(exc_type, exc_val, exc_tb)


class ToolServerLifecycleHandler:
    """Manages FastAPI server lifecycle for tool access."""

    def __init__(
        self,
        toolset_generator: ToolsetCodeGenerator,
        host: str = "localhost",
        port: int = 8000,
    ) -> None:
        self.toolset_generator = toolset_generator
        self.host = host
        self.port = port  # Will be set when socket is created
        self.app: FastAPI | None = None
        self.server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None

    async def __aenter__(self) -> ServerInfo:
        """Start FastAPI server with tool routes."""
        from exxec.models import ServerInfo
        from fastapi import FastAPI

        from llmling_agent.utils.network import _create_socket

        if self.server is not None:
            return ServerInfo(url=f"http://{self.host}:{self.port}", port=self.port)
        # Create socket and get actual port
        self._socket, self.port = _create_socket(self.port)
        # Create FastAPI app
        self.app = FastAPI(title="Tool Server", description="Generated tool endpoints")

        # Add tool routes
        self.toolset_generator.add_all_routes(self.app, "/tools")

        import uvicorn

        config = uvicorn.Config(
            self.app,
            log_level="error",  # Reduce log noise
            access_log=False,
            host=self.host,
            port=self.port,
            ws="websockets-sansio",
        )
        self.server = uvicorn.Server(config)

        # Start server in background with our socket
        self._server_task = asyncio.create_task(self.server.serve([self._socket]))
        logger.info("Started tool server", host=self.host, port=self.port)
        await asyncio.sleep(0.1)  # Wait for server to start properly
        # Verify server is running
        max_retries = 10
        for _ in range(max_retries):
            if self.server.started:
                break
            await asyncio.sleep(0.1)

        return ServerInfo(url=f"http://{self.host}:{self.port}", port=self.port)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop FastAPI server."""
        # Stop server gracefully
        if self.server:
            with contextlib.suppress(Exception):
                self.server.should_exit = True
                if hasattr(self.server, "force_exit"):
                    self.server.force_exit = True

        # Cancel server task
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._server_task, timeout=1.0)

        # Close socket
        if self._socket:
            with contextlib.suppress(Exception):
                self._socket.close()

        # Reset state
        self.server = None
        self._server_task = None
        self._socket = None
        self.app = None


if __name__ == "__main__":
    from exxec.configs import LocalExecutionEnvironmentConfig

    from llmling_agent.tools.base import Tool

    def add_numbers(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    async def main() -> None:
        tools = [Tool.from_callable(add_numbers)]
        config = LocalExecutionEnvironmentConfig()
        provider = RemoteCodeExecutor.from_tools(tools, config, server_port=9876)
        async with provider:
            result = await provider.execute_code("_result = await add_numbers(5, 3)")
            print(f"Result: {result.result}")

    asyncio.run(main())
