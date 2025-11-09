"""Orchestrates code generation for multiple tools."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers.codemode.namespace_callable import NamespaceCallable
from llmling_agent.resource_providers.codemode.tool_code_generator import (
    ToolCodeGenerator,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import FastAPI

    from llmling_agent.tools.base import Tool


USAGE = """\
Usage notes:
- Write your code inside an 'async def main():' function
- All tool functions are async, use 'await'
- Use 'return' statements to return values from main()
- Generated model classes are available for type checking
- Use 'await report_progress(current, total, message)' for long-running operations
- DO NOT call asyncio.run() or try to run the main function yourself
- DO NOT import asyncio or other modules - tools are already available
- Example:
    async def main():
        for i in range(5):
            await report_progress(i, 5, f'Step {i+1} for {name}')
            should_continue = await ask_user('Continue?', 'bool')
            if not should_continue:
                break
        return f'Completed for {name}'

"""


@dataclass
class ToolsetCodeGenerator:
    """Generates code artifacts for multiple tools."""

    generators: Sequence[ToolCodeGenerator]
    """ToolCodeGenerator instances for each tool."""

    include_signatures: bool = True
    """Include function signatures in documentation."""

    include_docstrings: bool = True
    """Include function docstrings in documentation."""

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[Tool],
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ) -> ToolsetCodeGenerator:
        """Create a ToolsetCodeGenerator from a sequence of Tools.

        Args:
            tools: Tools to generate code for
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation

        Returns:
            ToolsetCodeGenerator instance
        """
        generators = [ToolCodeGenerator.from_tool(tool) for tool in tools]
        return cls(generators, include_signatures, include_docstrings)

    def generate_tool_description(self) -> str:
        """Generate comprehensive tool description with available functions."""
        if not self.generators:
            return "Execute Python code (no tools available)"

        return_models = self.generate_return_models()
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

        for generator in self.generators:
            if self.include_signatures:
                signature = generator.get_function_signature()
                parts.append(f"async def {signature}:")
            else:
                parts.append(f"async def {generator.name}(...):")

            if self.include_docstrings and generator.callable.__doc__:
                indented_desc = "    " + generator.callable.__doc__.replace(
                    "\n", "\n    "
                )
                parts.append(f'    """{indented_desc}"""')
            parts.append("")

        parts.append(USAGE)

        return "\n".join(parts)

    def generate_execution_namespace(self) -> dict[str, Any]:
        """Build Python namespace with tool functions and generated models."""
        namespace: dict[str, Any] = {"__builtins__": __builtins__, "_result": None}

        # Add tool functions
        for generator in self.generators:
            namespace[generator.name] = NamespaceCallable.from_generator(generator)

        # Add generated model classes to namespace
        if models_code := self.generate_return_models():
            with contextlib.suppress(Exception):
                exec(models_code, namespace)

        return namespace

    def generate_return_models(self) -> str:
        """Generate Pydantic models for tool return types."""
        model_parts = [
            code for g in self.generators if (code := g.generate_return_model())
        ]
        return "\n\n".join(model_parts) if model_parts else ""

    def add_all_routes(self, app: FastAPI, path_prefix: str = "/tools") -> None:
        """Add FastAPI routes for all tools.

        Args:
            app: FastAPI application instance
            path_prefix: Path prefix for routes
        """
        for generator in self.generators:
            generator.add_route_to_app(app, path_prefix)


if __name__ == "__main__":
    import webbrowser

    from llmling_agent.tools.base import Tool

    t = Tool.from_callable(webbrowser.open)
    generator = ToolsetCodeGenerator.from_tools([t])
    models = generator.generate_return_models()
    print(models)
    namespace = generator.generate_execution_namespace()
    print(namespace)
