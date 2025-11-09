"""Secure code execution provider using isolated execution environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.resource_providers.codemode.code_execution_provider import (
    CodeExecutionProvider,
)
from llmling_agent.resource_providers.codemode.fix_code import fix_code
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent_config.execution_environments import ExecutionEnvironmentConfig


class SecureCodeModeResourceProvider(ResourceProvider):
    """Provider that executes code in secure isolation with tool access via server."""

    def __init__(
        self,
        wrapped_providers: Sequence[ResourceProvider] | None = None,
        wrapped_tools: Sequence[Tool] | None = None,
        execution_config: ExecutionEnvironmentConfig | None = None,
        name: str = "secure_code_executor",
        include_signatures: bool = True,
        include_docstrings: bool = True,
        server_host: str = "localhost",
        server_port: int = 8000,
    ):
        """Initialize secure code execution provider.

        Args:
            wrapped_providers: Providers whose tools to expose
            wrapped_tools: Individual tools to expose
            execution_config: Execution environment configuration
            name: Provider name
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation
            server_host: Host for tool server
            server_port: Port for tool server
        """
        super().__init__(name=name)
        self.wrapped_providers = list(wrapped_providers or [])
        self.wrapped_tools = list(wrapped_tools or [])
        self.execution_config = execution_config
        self.include_signatures = include_signatures
        self.include_docstrings = include_docstrings
        self.server_host = server_host
        self.server_port = server_port

        # Cache for expensive operations
        self._tools_cache: list[Tool] | None = None
        self._code_execution_provider: CodeExecutionProvider | None = None

    async def get_tools(self) -> list[Tool]:
        """Return single secure code execution tool."""
        code_provider = await self._get_code_execution_provider()
        description = code_provider.get_tool_description()

        return [
            Tool.from_callable(self.execute_secure_code, description_override=description)
        ]

    async def execute_secure_code(  # noqa: D417
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
        # Get code execution provider
        code_provider = await self._get_code_execution_provider()

        # Add progress reporting helper to code
        progress_helper = """
async def report_progress(current: int, total: int, message: str = "") -> None:
    '''Report progress during execution'''
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% ({current}/{total}) {message}")
"""

        # Fix and prepare code
        python_code = fix_code(python_code)
        full_code = f"{progress_helper}\n\n{python_code}"

        # Add context variables if provided (limited support in isolated environments)
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
                full_code = f"{progress_helper}\n{ctx_code}\n{python_code}"

        try:
            # Execute in secure environment
            async with code_provider:
                result = await code_provider.execute_code(full_code)

                if result.success:
                    if result.result is None:
                        return "Code executed successfully"
                    return result.result
                return f"Error executing code: {result.error}"

        except Exception as e:  # noqa: BLE001
            return f"Error in secure execution: {e!s}"

    async def execute_secure_code_stream(
        self,
        ctx: AgentContext,
        python_code: str,
        context_vars: dict[str, Any] | None = None,
    ):
        """Execute Python code with streaming output.

        Args:
            ctx: Agent context
            python_code: Python code to execute
            context_vars: Additional variables to make available

        Yields:
            Lines of output as they are produced
        """
        # Get code execution provider
        code_provider = await self._get_code_execution_provider()

        # Add progress reporting helper
        progress_helper = """
async def report_progress(current: int, total: int, message: str = "") -> None:
    '''Report progress during execution'''
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% ({current}/{total}) {message}")
"""

        # Fix and prepare code
        python_code = fix_code(python_code)
        full_code = f"{progress_helper}\n\n{python_code}"

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
                full_code = f"{progress_helper}\n{ctx_code}\n{python_code}"

        try:
            # Execute with streaming
            async with code_provider:
                async for line in code_provider.execute_code_stream(full_code):
                    yield line

        except Exception as e:  # noqa: BLE001
            yield f"Error in secure execution: {e!s}"

    async def _get_code_execution_provider(self) -> CodeExecutionProvider:
        """Get cached code execution provider."""
        if self._code_execution_provider is None:
            all_tools = await self._collect_all_tools()

            if self.execution_config is None:
                # Default to local execution if no config provided
                from llmling_agent_config.execution_environments import (
                    LocalExecutionEnvironmentConfig,
                )

                self.execution_config = LocalExecutionEnvironmentConfig()

            self._code_execution_provider = CodeExecutionProvider.from_tools_and_config(
                all_tools,
                self.execution_config,
                server_host=self.server_host,
                server_port=self.server_port,
                include_signatures=self.include_signatures,
                include_docstrings=self.include_docstrings,
            )

        return self._code_execution_provider

    async def _collect_all_tools(self) -> list[Tool]:
        """Collect all tools from providers and direct tools with caching."""
        if self._tools_cache is not None:
            return self._tools_cache

        all_tools = list(self.wrapped_tools)

        for provider in self.wrapped_providers:
            async with provider:
                provider_tools = await provider.get_tools()
            all_tools.extend(provider_tools)

        self._tools_cache = all_tools
        return all_tools

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup is handled by the code execution provider


if __name__ == "__main__":
    import asyncio

    from llmling_agent.resource_providers.static import StaticResourceProvider
    from llmling_agent_config.execution_environments import (
        LocalExecutionEnvironmentConfig,
    )

    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    async def main():
        # Create tools
        static_provider = StaticResourceProvider(
            tools=[
                Tool.from_callable(add_numbers),
                Tool.from_callable(greet),
            ]
        )

        # Create secure execution provider
        config = LocalExecutionEnvironmentConfig(timeout=30.0)
        provider = SecureCodeModeResourceProvider(
            wrapped_providers=[static_provider],
            execution_config=config,
            server_port=9999,
        )

        print("Available tools:")
        for tool in await provider.get_tools():
            print(f"- {tool.name}")
            print(f"  Description: {tool.description[:200]}...")

        # Test execution
        test_code = """
async def main():
    # Test tool access via HTTP
    result1 = await add_numbers(x=5, y=3)
    result2 = await greet(name="World", greeting="Hi")

    await report_progress(1, 2, "Called add_numbers")
    await report_progress(2, 2, "Called greet")

    return f"Results: {result1}, {result2}"
"""

        print("\n=== Testing secure code execution ===")

        # Create mock context
        class MockAgentContext:
            node_name = "test-agent"
            report_progress = None

        ctx = MockAgentContext()

        async with provider:
            result = await provider.execute_secure_code(ctx, test_code)
            print(f"Execution result: {result}")

    asyncio.run(main())
