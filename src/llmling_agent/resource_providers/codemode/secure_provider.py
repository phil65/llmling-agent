"""Secure code execution provider using isolated execution environments."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import AggregatingResourceProvider
from llmling_agent.resource_providers.codemode.code_execution_provider import (
    CodeExecutionProvider,
)
from llmling_agent.resource_providers.codemode.helpers import fix_code
from llmling_agent.tools.base import Tool
from llmling_agent_config.execution_environments import LocalExecutionEnvironmentConfig


if TYPE_CHECKING:
    from llmling_agent.resource_providers import ResourceProvider
    from llmling_agent_config.execution_environments import ExecutionEnvironmentConfig


PROGRESS_HELPER = """
async def report_progress(current: int, total: int, message: str = "") -> None:
    '''Report progress during execution'''
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% ({current}/{total}) {message}")
"""


class SecureCodeModeResourceProvider(AggregatingResourceProvider):
    """Provider that executes code in secure isolation with tool access via server."""

    def __init__(
        self,
        providers: list[ResourceProvider],
        execution_config: ExecutionEnvironmentConfig | None = None,
        name: str = "secure_code_executor",
        include_signatures: bool = True,
        include_docstrings: bool = True,
        server_host: str = "localhost",
        server_port: int = 8000,
    ):
        """Initialize secure code execution provider.

        Args:
            providers: Providers whose tools to expose
            execution_config: Execution environment configuration
            name: Provider name
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation
            server_host: Host for tool server
            server_port: Port for tool server
        """
        super().__init__(providers=list(providers), name=name)
        self.execution_config = execution_config or LocalExecutionEnvironmentConfig()
        self.include_signatures = include_signatures
        self.include_docstrings = include_docstrings
        self.server_host = server_host
        self.server_port = server_port
        self._tools_cache: list[Tool] | None = None
        self._code_execution_provider: CodeExecutionProvider | None = None
        self._provider_lock = asyncio.Lock()

    async def get_tools(self) -> list[Tool]:
        """Return single secure code execution tool."""
        code_provider = await self._get_code_execution_provider()
        desc = code_provider.get_tool_description()
        return [Tool.from_callable(self.execute, description_override=desc)]

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
                all_tools = await self._collect_all_tools()
                self._code_execution_provider = (
                    CodeExecutionProvider.from_tools_and_config(
                        all_tools,
                        self.execution_config,
                        server_host=self.server_host,
                        server_port=self.server_port,
                        include_signatures=self.include_signatures,
                        include_docstrings=self.include_docstrings,
                    )
                )

                # Initialize the provider and start server
                await self._code_execution_provider.__aenter__()

            return self._code_execution_provider

    async def _collect_all_tools(self) -> list[Tool]:
        """Collect all tools from providers and direct tools with caching."""
        if self._tools_cache is not None:
            return self._tools_cache

        all_tools = [t for provider in self.providers for t in await provider.get_tools()]
        self._tools_cache = all_tools
        return all_tools

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Clean up the code execution provider if it exists
        if self._code_execution_provider is not None:
            with contextlib.suppress(Exception):
                await self._code_execution_provider.__aexit__(exc_type, exc_val, exc_tb)
            self._code_execution_provider = None


if __name__ == "__main__":
    import asyncio
    import logging
    import sys
    import webbrowser

    from llmling_agent import Agent
    from llmling_agent.resource_providers import StaticResourceProvider
    from llmling_agent_config.execution_environments import (
        LocalExecutionEnvironmentConfig,
    )

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

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
