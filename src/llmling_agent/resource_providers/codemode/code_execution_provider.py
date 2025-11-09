"""Code execution provider with secure tool isolation via FastAPI server."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from llmling_agent.resource_providers.codemode.toolset_code_generator import (
    ToolsetCodeGenerator,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from anyenv.code_execution.base import ExecutionEnvironment
    from anyenv.code_execution.models import ServerInfo

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.execution_environments import ExecutionEnvironmentConfig


@dataclass
class CodeExecutionProvider:
    """Provides secure code execution with tool access via FastAPI server.

    Architecture:
    - FastAPI server runs in HOST environment with tool routes
    - User code runs in SANDBOX environment (Docker, E2B, etc.)
    - Sandbox makes HTTP calls to server for tool execution
    """

    toolset_generator: ToolsetCodeGenerator
    """Code generator for tools."""

    execution_environment: ExecutionEnvironment
    """Execution environment for running code."""

    @classmethod
    def from_tools_and_config(
        cls,
        tools: Sequence[Tool],
        env_config: ExecutionEnvironmentConfig,
        server_host: str = "localhost",
        server_port: int = 8000,
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ) -> CodeExecutionProvider:
        """Create provider from tools and environment configuration.

        Args:
            tools: Tools to make available for code execution
            env_config: Execution environment configuration
            server_host: Host for FastAPI server
            server_port: Port for FastAPI server
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation

        Returns:
            CodeExecutionProvider instance
        """
        # Create toolset code generator
        toolset_generator = ToolsetCodeGenerator.from_tools(
            tools, include_signatures, include_docstrings
        )

        # Create server lifecycle handler
        server_handler = ToolServerLifecycleHandler(
            toolset_generator, server_host, server_port
        )

        # Create execution environment with server lifecycle
        execution_env = env_config.get_provider(server_handler)

        return cls(toolset_generator, execution_env)

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
        async with self.execution_environment:
            return await self.execution_environment.execute(code)

    async def execute_code_stream(self, code: str):
        """Execute code and stream output.

        Args:
            code: Python code to execute

        Yields:
            Lines of output as they are produced
        """
        async with self.execution_environment:
            if hasattr(self.execution_environment, "execute_stream"):
                async for line in self.execution_environment.execute_stream(code):
                    yield line
            else:
                # Fallback to regular execution if streaming not supported
                result = await self.execution_environment.execute(code)
                yield str(result.result) if result.success else f"Error: {result.error}"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.execution_environment.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return await self.execution_environment.__aexit__(exc_type, exc_val, exc_tb)


class ToolServerLifecycleHandler:
    """Manages FastAPI server lifecycle for tool access."""

    def __init__(
        self,
        toolset_generator: ToolsetCodeGenerator,
        host: str = "localhost",
        port: int = 8000,
    ):
        self.toolset_generator = toolset_generator
        self.host = host
        self.port = port  # Will be set when socket is created
        self.app: FastAPI | None = None
        self.server: Any = None
        self._server_task: Any = None
        self._socket: Any = None

    def _create_socket(self, preferred_port: int):
        """Create a socket bound to a free port."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        return sock

    async def __aenter__(self) -> ServerInfo:
        """Start FastAPI server with tool routes."""
        # Create socket and get actual port
        self._socket = self._create_socket(self.port)

        # Create FastAPI app
        self.app = FastAPI(title="Tool Server", description="Generated tool endpoints")

        # Add tool routes
        self.toolset_generator.add_all_routes(self.app, "/tools")

        # Start server
        try:
            import uvicorn

            config = uvicorn.Config(self.app, log_level="warning")
            self.server = uvicorn.Server(config)

            # Start server in background with our socket
            import asyncio

            self._server_task = asyncio.create_task(self.server.serve([self._socket]))
            print(f"Started tool server on http://{self.host}:{self.port}")

            # Wait for server to start
            await asyncio.sleep(0.5)

        except ImportError:
            # Fallback if uvicorn not available
            pass

        # Return server info for execution environment
        from anyenv.code_execution.models import ServerInfo

        return ServerInfo(url=f"http://{self.host}:{self.port}", port=self.port, tools={})

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop FastAPI server."""
        if self.server:
            with contextlib.suppress(Exception):
                self.server.should_exit = True
                import asyncio

                await asyncio.sleep(0.1)

        if self._server_task and not self._server_task.done():
            with contextlib.suppress(Exception):
                self._server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._server_task

        if self._socket:
            with contextlib.suppress(Exception):
                self._socket.close()


if __name__ == "__main__":
    import asyncio

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.execution_environments import (
        LocalExecutionEnvironmentConfig,
    )

    def add_numbers(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    async def main():
        tools = [Tool.from_callable(add_numbers)]
        config = LocalExecutionEnvironmentConfig()
        provider = CodeExecutionProvider.from_tools_and_config(
            tools, config, server_port=9876
        )
        async with provider:
            result = await provider.execute_code("_result = await add_numbers(5, 3)")
            print(f"Result: {result.result}")

    asyncio.run(main())
