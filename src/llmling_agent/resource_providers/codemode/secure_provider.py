"""Secure code execution provider using isolated execution environments."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from anyenv.code_execution.configs import LocalExecutionEnvironmentConfig

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers.codemode.code_execution_provider import (
    CodeExecutionProvider,
)
from llmling_agent.resource_providers.codemode.helpers import fix_code
from llmling_agent.resource_providers.codemode.provider import (
    USAGE,
    CodeModeResourceProvider,
)
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from types import TracebackType

    from anyenv.code_execution.configs import ExecutionEnvironmentConfig

    from llmling_agent.resource_providers import ResourceProvider


PROGRESS_HELPER = """
async def report_progress(current: int, total: int, message: str = "") -> None:
    '''Report progress during execution'''
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% ({current}/{total}) {message}")
"""


class SecureCodeModeResourceProvider(CodeModeResourceProvider):
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
    ):
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
        self._code_execution_provider: CodeExecutionProvider | None = None
        self._provider_lock = asyncio.Lock()

    async def execute(  # noqa: D417
        self,
        ctx: AgentContext,
        python_code: str,
        context_vars: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code in secure environment with tools available via HTTP.

        Args:
            python_code: Python code to execute
            context_vars: Additional variables to make available (limited support)

        Returns:
            Result of the code execution
        """
        code_provider = await self._get_code_execution_provider()
        python_code = fix_code(python_code)
        full_code = f"{PROGRESS_HELPER}\n\n{python_code}"
        # Add context variables if provided
        if context_vars:
            ctx_assignments = []
            for key, value in context_vars.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    ctx_assignments.append(f"{key} = {value!r}")
                else:
                    ctx_assignments.append(
                        f"# {key} = <non-serializable: {type(value).__name__}>"
                    )

            if ctx_assignments:
                ctx_code = "# Context variables:\n" + "\n".join(ctx_assignments) + "\n\n"
                full_code = f"{PROGRESS_HELPER}\n{ctx_code}\n{python_code}"

        try:
            result = await code_provider.execution_environment.execute(full_code)
            if result.success:
                if result.result is None:
                    return "Code executed successfully"
                return result.result
        except Exception as e:  # noqa: BLE001
            return f"Error in secure execution: {e!s}"
        else:
            return f"Error executing code: {result.error}"

    async def _get_code_execution_provider(self) -> CodeExecutionProvider:
        """Get cached code execution provider with thread-safe initialization."""
        async with self._provider_lock:
            if self._code_execution_provider is None:
                all_tools = await super().get_tools()
                self._code_execution_provider = CodeExecutionProvider.from_tools(
                    all_tools,
                    self.execution_config,
                    server_host=self.server_host,
                    server_port=self.server_port,
                    include_docstrings=self.include_docstrings,
                )

                # Initialize the provider and start server
                await self._code_execution_provider.__aenter__()

            return self._code_execution_provider

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Async context manager exit."""
        # Clean up the code execution provider if it exists
        if self._code_execution_provider is not None:
            with contextlib.suppress(Exception):
                await self._code_execution_provider.__aexit__(exc_type, exc_val, exc_tb)
            self._code_execution_provider = None


if __name__ == "__main__":
    import asyncio
    import webbrowser

    from anyenv.code_execution.configs import LocalExecutionEnvironmentConfig

    from llmling_agent import Agent, log
    from llmling_agent.resource_providers import StaticResourceProvider

    log.configure_logging()

    def open_browser(url: str, new: int = 0, autoraise: bool = True) -> bool:
        """Display url using the default browser.

        If possible, open url in a location determined by new.
        - 0: the same browser window (the default).
        - 1: a new browser window.
        - 2: a new browser page ("tab").
        If possible, autoraise raises the window (the default) or not.

        If opening the browser succeeds, return True.
        If there is a problem, return False.
        """
        return webbrowser.open(url, new, autoraise)

    async def main():
        tools = [Tool.from_callable(open_browser)]
        static_provider = StaticResourceProvider(tools=tools)
        config = LocalExecutionEnvironmentConfig(timeout=30.0)
        provider = SecureCodeModeResourceProvider(
            providers=[static_provider],
            execution_config=config,
            server_port=9999,
        )

        print("Available tools:")
        for tool in await provider.get_tools():
            print(f"- {tool.name}: {tool.description[:100]}...")

        async with Agent(model="openai:gpt-5-nano") as agent:
            agent.tools.add_provider(provider)
            result = await agent.run(
                "Use code execution to open google.com in the browser."
            )
            print(f"Result: {result}")

    asyncio.run(main())
