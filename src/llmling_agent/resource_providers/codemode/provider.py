"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import AggregatingResourceProvider
from llmling_agent.resource_providers.codemode.default_prompt import USAGE
from llmling_agent.resource_providers.codemode.helpers import (
    tools_to_codegen,
    validate_code,
)
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from schemez import ToolsetCodeGenerator

    from llmling_agent.resource_providers import ResourceProvider


class CodeModeResourceProvider(AggregatingResourceProvider):
    """Provider that wraps tools into a single Python execution environment."""

    def __init__(
        self,
        providers: list[ResourceProvider],
        name: str = "meta_tools",
        include_docstrings: bool = True,
        usage_notes: str = USAGE,
    ) -> None:
        """Initialize meta provider.

        Args:
            providers: Providers whose tools to wrap
            name: Provider name
            include_docstrings: Include function docstrings in documentation
            usage_notes: Usage notes for the codemode tool
        """
        super().__init__(providers=providers, name=name)
        self.include_docstrings = include_docstrings
        self._toolset_generator: ToolsetCodeGenerator | None = None
        self._cached_tool: Tool | None = None
        self.usage_notes = usage_notes

    async def get_tools(self) -> list[Tool]:
        """Return single meta-tool for Python execution with available tools."""
        # Always generate fresh toolset to reflect current tools
        toolset_generator = await self._get_fresh_code_generator()
        desc = toolset_generator.generate_tool_description()
        desc += self.usage_notes

        if self._cached_tool is None:
            # Create a closure that captures self but isn't a bound method
            async def execute_tool(
                ctx: AgentContext,
                python_code: str,
            ) -> Any:
                """These docstings are overriden by description_override."""
                return await self.execute(ctx, python_code)

            self._cached_tool = Tool.from_callable(execute_tool, description_override=desc)
        else:
            # Update the description on existing cached tool
            self._cached_tool.description = desc

        return [self._cached_tool]

    async def execute(self, ctx: AgentContext, python_code: str) -> Any:  # noqa: D417
        """Execute Python code with all wrapped tools available as functions.

        Args:
            python_code: Python code to execute

        Returns:
            Result of the last expression or explicit return value
        """
        toolset_generator = await self._get_fresh_code_generator()
        namespace = toolset_generator.generate_execution_namespace()

        # async def report_progress(current: int, total: int, message: str = ""):
        #     """Report progress during code execution."""
        #     await ctx.report_progress(current, total, message)

        # namespace["report_progress"] = NamespaceCallable(report_progress)

        validate_code(python_code)
        try:
            exec(python_code, namespace)
            result = await namespace["main"]()
            # Handle edge cases with coroutines and return values
            if inspect.iscoroutine(result):
                result = await result
            if not result:  # in order to not confuse the model, return a success message.
                return "Code executed successfully"
        except Exception as e:  # noqa: BLE001
            return f"Error executing code: {e!s}"
        else:
            return result

    def invalidate_cache(self) -> None:
        """Invalidate cached tool when providers change."""
        self._cached_tool = None
        # Note: We no longer cache the toolset generator, so no need to clear it

    async def _get_fresh_code_generator(self) -> ToolsetCodeGenerator:
        """Get fresh toolset generator with current tools."""
        return tools_to_codegen(
            tools=await super().get_tools(),
            include_docstrings=self.include_docstrings,
        )


if __name__ == "__main__":
    import asyncio
    import webbrowser

    from llmling_agent import Agent, log
    from llmling_agent.resource_providers import StaticResourceProvider

    log.configure_logging()
    static_provider = StaticResourceProvider(tools=[Tool.from_callable(webbrowser.open)])
    provider = CodeModeResourceProvider([static_provider])

    async def main() -> None:
        print("Available tools:")
        for tool in await provider.get_tools():
            print(f"- {tool.name}: {tool.description}")

        async with Agent(model="openai:gpt-5-nano") as agent:
            agent.tools.add_provider(provider)
            result = await agent.run("Open google.com in a new tab.")
            print(f"Result: {result}")

    asyncio.run(main())
