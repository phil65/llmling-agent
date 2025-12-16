"""Secure code execution provider using isolated execution environments."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Self

from exxec.configs import LocalExecutionEnvironmentConfig

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.codemode.code_executor import (
    RemoteCodeExecutor,
)
from llmling_agent.resource_providers.codemode.default_prompt import USAGE
from llmling_agent.resource_providers.codemode.helpers import validate_code
from llmling_agent.resource_providers.codemode.provider import (
    CodeModeResourceProvider,
)
from llmling_agent.tools.base import Tool


logger = get_logger(__name__)

if TYPE_CHECKING:
    from types import TracebackType

    from exxec.configs import ExecutionEnvironmentConfig

    from llmling_agent.resource_providers import ResourceProvider


PROGRESS_HELPER = """
async def report_progress(current: int, total: int, message: str = "") -> None:
    '''Report progress during execution'''
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% ({current}/{total}) {message}")
"""


class RemoteCodeModeResourceProvider(CodeModeResourceProvider):
    """Provider that executes code in secure isolation with tool access via server."""

    def __init__(
        self,
        providers: list[ResourceProvider],
        execution_config: ExecutionEnvironmentConfig | None = None,
        name: str = "secure_code_executor",
        include_docstrings: bool = True,
        usage_notes: str = USAGE,
        server_host: str = "localhost",
        server_port: int = 8000,
    ) -> None:
        """Initialize secure code execution provider.

        Args:
            providers: Providers whose tools to expose
            execution_config: Execution environment configuration
            name: Provider name
            include_docstrings: Include function docstrings in documentation
            usage_notes: Usage notes for the provider
            server_host: Host for tool server
            server_port: Port for tool server
        """
        super().__init__(
            providers=providers,
            name=name,
            include_docstrings=include_docstrings,
            usage_notes=usage_notes,
        )
        self.execution_config = execution_config or LocalExecutionEnvironmentConfig()
        self.server_host = server_host
        self.server_port = server_port
        self._code_executor: RemoteCodeExecutor | None = None
        self._provider_lock = asyncio.Lock()

    async def execute(self, ctx: AgentContext, python_code: str) -> Any:  # noqa: D417
        """Execute Python code in secure environment with tools available via HTTP.

        Args:
            python_code: Python code to execute

        Returns:
            Result of the code execution
        """
        code_provider = await self._get_code_executor()
        logger.info("Validating code", code=python_code)
        validate_code(python_code)
        full_code = f"{PROGRESS_HELPER}\n\n{python_code}"
        logger.info("Complete code", code=full_code)
        try:
            result = await code_provider.execution_env.execute(full_code)
            if result.success:
                logger.info("Code executed successfully")
                if result.result is None:
                    return "Code executed successfully"
                return result.result
        except Exception as e:  # noqa: BLE001
            return f"Error in secure execution: {e!s}"
        else:
            return f"Error executing code: {result.error}"

    async def _get_code_executor(self) -> RemoteCodeExecutor:
        """Get cached code execution provider with thread-safe initialization."""
        async with self._provider_lock:
            if self._code_executor is None:
                all_tools = await super().get_tools()
                self._code_executor = RemoteCodeExecutor.from_tools(
                    all_tools,
                    self.execution_config,
                    server_host=self.server_host,
                    server_port=self.server_port,
                    include_docstrings=self.include_docstrings,
                )
                # Initialize the provider and start server
                await self._code_executor.__aenter__()

            return self._code_executor

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        if self._code_executor is not None:
            with contextlib.suppress(Exception):
                await self._code_executor.__aexit__(exc_type, exc_val, exc_tb)
            self._code_executor = None


if __name__ == "__main__":
    import webbrowser

    from exxec.configs import LocalExecutionEnvironmentConfig

    from llmling_agent import Agent, log
    from llmling_agent.resource_providers import StaticResourceProvider

    log.configure_logging()

    def open_browser(url: str) -> bool:
        """Use this to open url in the default browser."""
        return webbrowser.open(url)

    async def main() -> None:
        tools = [Tool.from_callable(open_browser)]
        static_provider = StaticResourceProvider(tools=tools)
        config = LocalExecutionEnvironmentConfig(timeout=30.0)
        provider = RemoteCodeModeResourceProvider(
            providers=[static_provider],
            execution_config=config,
            server_port=9999,
        )
        print("Available tools:")
        for tool in await provider.get_tools():
            print(f"- {tool.name}: {tool.description}")

        async with Agent(model="anthropic:claude-haiku-4-5") as agent:
            agent.tools.add_provider(provider)
            result = await agent.run("open google.com in the browser.")
            print(f"Result: {result}")

    asyncio.run(main())
