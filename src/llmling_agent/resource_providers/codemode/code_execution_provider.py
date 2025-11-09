"""Code execution provider that combines tool code generation with execution environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers.codemode.toolset_code_generator import (
    ToolsetCodeGenerator,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from anyenv.code_execution.base import ExecutionEnvironment
    from fastapi import FastAPI

    from llmling_agent.tools.base import Tool
    from llmling_agent_config.execution_environments import ExecutionEnvironmentConfig


@dataclass
class CodeExecutionProvider:
    """Provides code execution capabilities with tool integration.

    Combines tool code generation with configurable execution environments
    to provide a complete code execution solution.
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
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ) -> CodeExecutionProvider:
        """Create provider from tools and environment configuration.

        Args:
            tools: Tools to make available for code execution
            env_config: Execution environment configuration
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation

        Returns:
            CodeExecutionProvider instance
        """
        # Create toolset code generator
        toolset_generator = ToolsetCodeGenerator.from_tools(
            tools, include_signatures, include_docstrings
        )

        # Create execution environment from config
        execution_env = _create_execution_environment(env_config)

        return cls(toolset_generator, execution_env)

    @classmethod
    def from_tools_and_environment(
        cls,
        tools: Sequence[Tool],
        execution_environment: ExecutionEnvironment,
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ) -> CodeExecutionProvider:
        """Create provider from tools and execution environment instance.

        Args:
            tools: Tools to make available for code execution
            execution_environment: Pre-configured execution environment
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation

        Returns:
            CodeExecutionProvider instance
        """
        toolset_generator = ToolsetCodeGenerator.from_tools(
            tools, include_signatures, include_docstrings
        )

        return cls(toolset_generator, execution_environment)

    def get_tool_description(self) -> str:
        """Get comprehensive description of available tools and execution environment."""
        base_description = self.toolset_generator.generate_tool_description()

        # Add execution environment info
        env_info = f"\nExecution Environment: {type(self.execution_environment).__name__}"

        return base_description + env_info

    async def execute_code(self, code: str) -> Any:
        """Execute code with tools available in the namespace.

        Args:
            code: Python code to execute

        Returns:
            Execution result from the environment
        """
        # Get the execution namespace with tools
        namespace = self.toolset_generator.generate_execution_namespace()

        # Inject namespace into the execution environment if supported
        if hasattr(self.execution_environment, "set_namespace"):
            self.execution_environment.set_namespace(namespace)

        # For environments that don't support namespace injection,
        # we need to modify the code to include tool definitions
        if not hasattr(self.execution_environment, "set_namespace"):
            # Prepend tool definitions to the code
            tool_code = self._generate_tool_code_prefix()
            full_code = f"{tool_code}\n\n{code}"
        else:
            full_code = code

        async with self.execution_environment:
            return await self.execution_environment.execute(full_code)

    async def execute_code_stream(self, code: str):
        """Execute code with tools and stream output.

        Args:
            code: Python code to execute

        Yields:
            Lines of output as they are produced
        """
        # Get the execution namespace with tools
        namespace = self.toolset_generator.generate_execution_namespace()

        # Inject namespace into the execution environment if supported
        if hasattr(self.execution_environment, "set_namespace"):
            self.execution_environment.set_namespace(namespace)

        # For environments that don't support namespace injection,
        # we need to modify the code to include tool definitions
        if not hasattr(self.execution_environment, "set_namespace"):
            # Prepend tool definitions to the code
            tool_code = self._generate_tool_code_prefix()
            full_code = f"{tool_code}\n\n{code}"
        else:
            full_code = code

        async with self.execution_environment:
            if hasattr(self.execution_environment, "execute_stream"):
                async for line in self.execution_environment.execute_stream(full_code):
                    yield line
            else:
                # Fallback to regular execution if streaming not supported
                result = await self.execution_environment.execute(full_code)
                yield str(result.result) if result.success else f"Error: {result.error}"

    def add_routes_to_app(self, app: FastAPI, path_prefix: str = "/tools") -> None:
        """Add FastAPI routes for all tools to the app.

        Args:
            app: FastAPI application instance
            path_prefix: Path prefix for routes
        """
        self.toolset_generator.add_all_routes(app, path_prefix)

    def _generate_tool_code_prefix(self) -> str:
        """Generate code that defines tool functions for injection into execution.

        This is used for execution environments that don't support namespace injection.

        Returns:
            Python code that defines all tool functions
        """
        code_parts = []

        # Add necessary imports
        code_parts.append("import asyncio")
        code_parts.append("from typing import Any")
        code_parts.append("")

        # Add tool function definitions
        for generator in self.toolset_generator.generators:
            func_name = generator.name

            # Create a wrapper function that calls the actual tool
            wrapper_code = f"""
async def {func_name}(*args, **kwargs) -> Any:
    '''Tool wrapper for {func_name}'''
    try:
        # This would need to be implemented to actually call the tool
        # For now, return a placeholder
        return f"Called {func_name} with args={{args}} kwargs={{kwargs}}"
    except Exception as e:
        return f"Error in {func_name}: {{e}}"
"""
            code_parts.append(wrapper_code)

        # Add helper functions
        helper_code = """
async def report_progress(current: int, total: int, message: str = "") -> None:
    '''Report progress during execution'''
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {percentage:.1f}% ({current}/{total}) {message}")

async def ask_user(message: str, response_type: str = "string") -> Any:
    '''Ask user for input (placeholder implementation)'''
    print(f"User input requested: {message} (type: {response_type})")
    return f"mock_{response_type}_response"
"""
        code_parts.append(helper_code)

        return "\n".join(code_parts)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.execution_environment.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return await self.execution_environment.__aexit__(exc_type, exc_val, exc_tb)


def _create_execution_environment(
    config: ExecutionEnvironmentConfig,
) -> ExecutionEnvironment:
    """Create execution environment from configuration.

    Args:
        config: Execution environment configuration

    Returns:
        Configured execution environment instance

    Raises:
        ValueError: If environment type is not supported
    """
    if config.type == "local":
        from anyenv.code_execution.local_provider import LocalExecutionEnvironment

        return LocalExecutionEnvironment(
            dependencies=config.dependencies,
            timeout=config.timeout,
        )

    if config.type == "subprocess":
        from anyenv.code_execution.subprocess_provider import (
            SubprocessExecutionEnvironment,
        )

        return SubprocessExecutionEnvironment(
            dependencies=config.dependencies,
            executable=config.executable,
            timeout=config.timeout,
            language=config.language,
        )

    if config.type == "docker":
        from anyenv.code_execution.docker_provider import DockerExecutionEnvironment

        return DockerExecutionEnvironment(
            dependencies=config.dependencies,
            image=config.image,
            timeout=config.timeout,
            language=config.language,
        )

    if config.type == "e2b":
        from anyenv.code_execution.e2b_provider import E2bExecutionEnvironment

        return E2bExecutionEnvironment(
            dependencies=config.dependencies,
            template=config.template,
            timeout=config.timeout,
            keep_alive=config.keep_alive,
            language=config.language,
        )

    if config.type == "beam":
        from anyenv.code_execution.beam_provider import BeamExecutionEnvironment

        return BeamExecutionEnvironment(
            dependencies=config.dependencies,
            cpu=config.cpu,
            memory=config.memory,
            keep_warm_seconds=config.keep_warm_seconds,
            timeout=config.timeout,
            language=config.language,
        )

    if config.type == "daytona":
        from anyenv.code_execution.daytona_provider import DaytonaExecutionEnvironment

        api_key_str = config.api_key.get_secret_value() if config.api_key else None

        return DaytonaExecutionEnvironment(
            dependencies=config.dependencies,
            api_url=config.api_url,
            api_key=api_key_str,
            target=config.target,
            image=config.image,
            timeout=config.timeout,
            keep_alive=config.keep_alive,
        )

    if config.type == "mcp_python":
        from anyenv.code_execution.mcp_python_provider import (
            McpPythonExecutionEnvironment,
        )

        return McpPythonExecutionEnvironment(
            dependencies=config.dependencies,
            allow_networking=config.allow_networking,
            timeout=config.timeout,
        )

    raise ValueError(f"Unsupported execution environment type: {config.type}")


if __name__ == "__main__":
    import asyncio

    from llmling_agent.tools.base import Tool
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
        tools = [
            Tool.from_callable(add_numbers),
            Tool.from_callable(greet),
        ]

        # Create execution environment config
        env_config = LocalExecutionEnvironmentConfig()

        # Create code execution provider
        provider = CodeExecutionProvider.from_tools_and_config(tools, env_config)

        print("Tool Description:")
        print(provider.get_tool_description())
        print("\n" + "=" * 50 + "\n")

        # Test code execution
        test_code = """
async def main():
    result1 = await add_numbers(5, 3)
    result2 = await greet("World", "Hi")
    return f"Results: {result1}, {result2}"
"""

        async with provider:
            result = await provider.execute_code(test_code)
            print(f"Execution result: {result}")

    asyncio.run(main())
