"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from functools import partial
import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import AggregatingResourceProvider
from llmling_agent.resource_providers.codemode.default_prompt import USAGE
from llmling_agent.resource_providers.codemode.helpers import (
    tools_to_codegen,
    validate_code,
)
from llmling_agent_toolsets.fsspec_toolset.toolset import FSSpecTools


if TYPE_CHECKING:
    from schemez import ToolsetCodeGenerator

    from llmling_agent.resource_providers import ResourceProvider
    from llmling_agent.tools.base import Tool


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
        self._cached_tool: Tool | None = None
        self.usage_notes = usage_notes

    async def get_tools(self) -> list[Tool]:
        """Return single meta-tool for Python execution with available tools."""
        # Always generate fresh toolset to reflect current tools
        toolset_generator = await self._get_code_generator()
        desc = toolset_generator.generate_tool_description()
        desc += self.usage_notes

        if self._cached_tool is None:
            # Create a closure that captures self but isn't a bound method
            async def execute_tool(ctx: AgentContext, python_code: str) -> Any:
                """These docstings are overriden by description_override."""
                return await self.execute(ctx, python_code)

            self._cached_tool = self.create_tool(execute_tool, description_override=desc)
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
        toolset_generator = await self._get_code_generator()
        namespace = toolset_generator.generate_execution_namespace()

        # Wrap namespace callables to inject AgentContext
        for value in namespace.values():
            if callable(value) and hasattr(value, "callable"):
                # It's a NamespaceCallable - wrap its underlying callable with ctx
                original_callable = value.callable
                if "agent_ctx" in inspect.signature(original_callable).parameters:
                    value.callable = partial(original_callable, agent_ctx=ctx)

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

    async def _get_code_generator(self) -> ToolsetCodeGenerator:
        """Get fresh toolset generator with current tools."""
        tools = await super().get_tools()
        return tools_to_codegen(tools=tools, include_docstrings=self.include_docstrings)


if __name__ == "__main__":
    import asyncio

    from llmling_agent import Agent
    from llmling_agent.delegation.pool import AgentPool

    static_provider = FSSpecTools()
    provider = CodeModeResourceProvider([static_provider])

    async def main() -> None:
        print("Available tools:")
        for tool in await provider.get_tools():
            print(f"- {tool.name}: {tool.description[:100]}...")

        async with AgentPool() as pool:
            agent: Agent[None, str] = Agent(
                model="openai:gpt-5-nano", event_handlers=["simple"], retries=1
            )
            pool.register("test_agent", agent)
            async with agent:
                agent.tools.add_provider(provider)
                prompt = (
                    "Call list_directory with path='.'. "
                    "Write: async def main(): "
                    "result = await list_directory(path='.'); "
                    "return result"
                )
                result = await agent.run(prompt)
                print(f"Result: {result}")

    asyncio.run(main())
