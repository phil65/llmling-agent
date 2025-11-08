"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.resource_providers.codemode import CodeGenerator
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
        self._description_cache: str | None = None
        self._namespace_cache: dict[str, Any] | None = None
        self._models_code_cache: str | None = None

    async def get_tools(self) -> list[Tool]:
        """Return single meta-tool for Python execution with available tools."""
        if self._description_cache is None:
            self._description_cache = await self._build_tool_description()

        return [
            Tool.from_callable(
                self.execute_codemode, description_override=self._description_cache
            )
        ]

    async def execute_codemode(
        self, python_code: str, context_vars: dict[str, Any] | None = None
    ) -> Any:
        """Execute Python code with all wrapped tools available as functions.

        Args:
            python_code: Python code to execute
            context_vars: Additional variables to make available

        Returns:
            Result of the last expression or explicit return value
        """
        # Build execution namespace with caching
        if self._namespace_cache is None:
            self._namespace_cache = await self._build_execution_namespace()

        # Create a copy to avoid modifying the cache
        namespace = self._namespace_cache.copy()
        if context_vars:
            namespace.update(context_vars)

        # Simplified execution: require main() function pattern
        if "async def main(" not in python_code:
            # Auto-wrap code in main function
            python_code = f"""async def main():
{chr(10).join("    " + line for line in python_code.splitlines())}"""

        exec(python_code, namespace)
        return await namespace["main"]()

    async def _build_tool_description(self) -> str:
        """Generate comprehensive tool description with available functions."""
        all_tools = await self._collect_all_tools()

        if not all_tools:
            return "Execute Python code (no tools available)"

        # Generate return type models if available
        return_models = _generate_return_models(all_tools)

        parts = [
            "Execute Python code with the following tools available as async functions:",
            "",
        ]

        if return_models:
            parts.extend([
                "# Generated return type models",
                return_models,
                "",
                "# Available functions:",
                "",
            ])

        for tool in all_tools:
            if self.include_signatures:
                signature = CodeGenerator.from_tool(tool).get_function_signature()
                parts.append(f"async def {signature}:")
            else:
                parts.append(f"async def {tool.name}(...):")

            if self.include_docstrings and tool.description:
                indented_desc = "    " + tool.description.replace("\n", "\n    ")
                parts.append(f'    """{indented_desc}"""')
            parts.append("")

        parts.extend([
            "Usage notes:",
            "- Write your code inside an 'async def main():' function",
            "- All tool functions are async, use 'await'",
            "- Use 'return' statements to return values from main()",
            "- Generated model classes are available for type checking",
            "- DO NOT call asyncio.run() or try to run the main function yourself",
            "- DO NOT import asyncio or other modules - tools are already available",
            "- Example:",
            "    async def main():",
            "        result = await open(url='https://example.com', new=2)",
            "        return result",
        ])

        return "\n".join(parts)

    async def _build_execution_namespace(self) -> dict[str, Any]:
        """Build Python namespace with tool functions and generated models."""
        namespace = {
            "__builtins__": __builtins__,
            "_result": None,
        }

        # Add tool functions
        for tool in await self._collect_all_tools():

            def make_tool_func(t: Tool):
                async def tool_func(*args, **kwargs):
                    return await t.execute(*args, **kwargs)

                tool_func.__name__ = t.name
                tool_func.__doc__ = t.description
                return tool_func

            namespace[tool.name] = make_tool_func(tool)

        # Add generated model classes to namespace
        if self._models_code_cache is None:
            self._models_code_cache = _generate_return_models(
                await self._collect_all_tools()
            )

        if self._models_code_cache:
            with contextlib.suppress(Exception):
                exec(self._models_code_cache, namespace)
        return namespace

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


def _generate_return_models(tools: list[Tool]) -> str:
    """Generate Pydantic models for tool return types."""
    model_parts = [
        code
        for t in tools
        if (code := CodeGenerator.from_tool(t).generate_return_model())
    ]
    return "\n\n".join(model_parts) if model_parts else ""


if __name__ == "__main__":
    import asyncio
    import logging
    import sys
    import webbrowser

    from llmling_agent import Agent
    from llmling_agent.resource_providers.static import StaticResourceProvider

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    static_provider = StaticResourceProvider(tools=[Tool.from_callable(webbrowser.open)])

    async def main():
        provider = CodeModeResourceProvider([static_provider])
        async with Agent(model="openai:gpt-4o-mini") as agent:
            agent.tools.add_provider(provider)
            result = await agent.run(
                "Use the available open() function to open a web browser "
                "with URL https://www.google.com."
            )
            print(result)

    asyncio.run(main())
