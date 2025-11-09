"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.resource_providers.codemode.fix_code import fix_code
from llmling_agent.resource_providers.codemode.namespace_callable import (
    NamespaceCallable,
)
from llmling_agent.resource_providers.codemode.toolset_code_generator import (
    ToolsetCodeGenerator,
)
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Sequence


class CodeModeResourceProvider(ResourceProvider):
    """Provider that wraps tools into a single Python execution environment."""

    def __init__(
        self,
        wrapped_providers: Sequence[ResourceProvider] | None = None,
        wrapped_tools: Sequence[Tool] | None = None,
        name: str = "meta_tools",
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ):
        """Initialize meta provider.

        Args:
            wrapped_providers: Providers whose tools to wrap
            wrapped_tools: Individual tools to wrap
            name: Provider name
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation
        """
        super().__init__(name=name)
        self.wrapped_providers = list(wrapped_providers or [])
        self.wrapped_tools = list(wrapped_tools or [])
        self.include_signatures = include_signatures
        self.include_docstrings = include_docstrings

        # Cache for expensive operations
        self._tools_cache: list[Tool] | None = None
        self._toolset_generator: ToolsetCodeGenerator | None = None

    async def get_tools(self) -> list[Tool]:
        """Return single meta-tool for Python execution with available tools."""
        toolset_generator = await self._get_toolset_generator()
        description = toolset_generator.generate_tool_description()

        return [
            Tool.from_callable(self.execute_codemode, description_override=description)
        ]

    async def execute_codemode(  # noqa: D417
        self,
        ctx: AgentContext,
        python_code: str,
        context_vars: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code with all wrapped tools available as functions.

        Args:
            python_code: Python code to execute
            context_vars: Additional variables to make available

        Returns:
            Result of the last expression or explicit return value
        """
        # Handle RunContext wrapper
        # if isinstance(ctx, RunContext):
        #     ctx = ctx.deps
        # Build execution namespace
        toolset_generator = await self._get_toolset_generator()
        namespace = toolset_generator.generate_execution_namespace()

        # Add progress reporting if context is available
        if ctx.report_progress:

            async def report_progress(current: int, total: int, message: str = ""):
                """Report progress during code execution."""
                assert ctx.report_progress
                await ctx.report_progress(current, total, message)

            namespace["report_progress"] = NamespaceCallable(
                report_progress,
            )

        # async def ask_user(
        #     message: str, response_type: str = "string"
        # ) -> str | bool | int | float | dict:
        #     """Ask the user for input during code execution.

        #     Args:
        #         message: Question to ask the user
        #         response_type: Type of response
        #                         expected ("string", "bool", "int", "float", "json")

        #     Returns:
        #         User's response in the requested type
        #     """
        #     from mcp import types

        #     # Map string types to Python types for elicitation
        #     type_mapping = {
        #         "string": str,
        #         "str": str,
        #         "bool": bool,
        #         "boolean": bool,
        #         "int": int,
        #         "integer": int,
        #         "float": float,
        #         "number": float,
        #         "json": dict,
        #         "dict": dict,
        #     }

        #     python_type = type_mapping.get(response_type.lower(), str)

        #     params = types.ElicitRequestParams(
        #         message=message,
        #         response_type=python_type.__name__,
        #     )

        #     result = await ctx.handle_elicitation(params)

        #     if isinstance(result, types.ElicitResult) and result.action == "accept":
        #         return result.content if result.content is not None else ""
        #     if isinstance(result, types.ErrorData):
        #         msg = f"Elicitation failed: {result.message}"
        #         raise RuntimeError(msg)
        #     msg = "User declined to provide input"
        #     raise RuntimeError(msg)

        # namespace["ask_user"] = ask_user

        if context_vars:
            namespace.update(context_vars)
        python_code = fix_code(python_code)
        try:
            exec(python_code, namespace)
            result = await namespace["main"]()

            # Handle edge cases with coroutines and return values
            if inspect.iscoroutine(result):
                result = await result

            # Ensure we return a serializable value
            if result is None:
                return "Code executed successfully"
            if hasattr(result, "__dict__") and not isinstance(
                result, (str, int, float, bool, list, dict)
            ):
                # Handle complex objects that might not serialize well
                return f"Operation completed. Result type: {type(result).__name__}"

        except Exception as e:  # noqa: BLE001
            return f"Error executing code: {e!s}"
        else:
            return result

    async def _get_toolset_generator(self) -> ToolsetCodeGenerator:
        """Get cached toolset generator."""
        if self._toolset_generator is None:
            all_tools = await self._collect_all_tools()
            self._toolset_generator = ToolsetCodeGenerator.from_tools(
                all_tools,
                include_signatures=self.include_signatures,
                include_docstrings=self.include_docstrings,
            )
        return self._toolset_generator

    async def _collect_all_tools(self) -> list[Tool]:
        """Collect all tools from providers and direct tools with caching."""
        if self._tools_cache is not None:
            return self._tools_cache

        all_tools = list(self.wrapped_tools)

        for provider in self.wrapped_providers:
            async with provider:
                provider_tools = await provider.get_tools()
            all_tools.extend(provider_tools)

        # Validate tools for common async issues
        validated_tools = []
        for tool in all_tools:
            if inspect.iscoroutinefunction(tool.callable):
                # Check if async function has proper return type hints
                sig = inspect.signature(tool.callable)
                if sig.return_annotation == inspect.Signature.empty:
                    # Add warning in tool description about missing return type
                    tool.description = f"{tool.description}\n\nNote: This async function should explicitly return a value."  # noqa: E501
            validated_tools.append(tool)

        self._tools_cache = validated_tools
        return validated_tools


if __name__ == "__main__":
    import asyncio
    import logging
    import sys
    import webbrowser

    from llmling_agent import Agent
    from llmling_agent.resource_providers.static import StaticResourceProvider

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    static_provider = StaticResourceProvider(tools=[Tool.from_callable(webbrowser.open)])

    async def main():
        provider = CodeModeResourceProvider([static_provider])
        print("Available tools:")
        for tool in await provider.get_tools():
            print(f"- {tool.name}: {tool.description[:100]}...")

        async with Agent(model="openai:gpt-5-nano") as agent:
            agent.tools.add_provider(provider)

            # Test elicitation functionality
            print("\n=== Testing Elicitation (User Input) ===")
            result = await agent.run("Open google.com in a new tab.")
            print(f"Result: {result}")

    asyncio.run(main())
