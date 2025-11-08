"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers import ResourceProvider
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
        # Build execution namespace
        toolset_generator = await self._get_toolset_generator()
        namespace = toolset_generator.generate_execution_namespace()

        if context_vars:
            namespace.update(context_vars)

        # Simplified execution: require main() function pattern
        if "async def main(" not in python_code:
            # Auto-wrap code in main function, ensuring last expression is returned
            lines = python_code.strip().splitlines()
            if lines:
                # Check if last line is an expression (not a statement)
                last_line = lines[-1].strip()
                if last_line and not any(
                    last_line.startswith(kw)
                    for kw in [
                        "import ",
                        "from ",
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "try ",
                        "with ",
                        "async def ",
                    ]
                ):
                    # Last line looks like an expression, add return
                    lines[-1] = f"    return {last_line}"
                    indented_lines = [f"    {line}" for line in lines[:-1]] + [lines[-1]]
                else:
                    indented_lines = [f"    {line}" for line in lines]
                python_code = "async def main():\n" + "\n".join(indented_lines)
            else:
                python_code = "async def main():\n    pass"

        try:
            exec(python_code, namespace)
            result = await namespace["main"]()
            # Ensure we return something meaningful instead of None
        except Exception as e:  # noqa: BLE001
            return f"Error executing code: {e!s}"
        else:
            return result if result is not None else "Code executed successfully"

    async def _get_toolset_generator(self) -> ToolsetCodeGenerator:
        """Get cached toolset generator."""
        if self._toolset_generator is None:
            all_tools = await self._collect_all_tools()
            self._toolset_generator = ToolsetCodeGenerator(
                tools=all_tools,
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

        self._tools_cache = all_tools
        return all_tools


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
