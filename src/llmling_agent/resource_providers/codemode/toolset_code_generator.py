"""Orchestrates code generation for multiple tools."""

from __future__ import annotations

import contextlib
import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers.codemode.tool_code_generator import (
    ToolCodeGenerator,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.tools.base import Tool


class ToolsetCodeGenerator:
    """Generates code artifacts for multiple tools."""

    def __init__(
        self,
        tools: Sequence[Tool],
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ):
        """Initialize toolset generator.

        Args:
            tools: Tools to generate code for
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation
        """
        self.tools = list(tools)
        self.include_signatures = include_signatures
        self.include_docstrings = include_docstrings

    def generate_tool_description(self) -> str:
        """Generate comprehensive tool description with available functions."""
        if not self.tools:
            return "Execute Python code (no tools available)"

        # Generate return type models if available
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

        for tool in self.tools:
            if self.include_signatures:
                generator = ToolCodeGenerator.from_tool(tool)
                signature = generator.get_function_signature()
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
            "- Use 'await report_progress(current, total, message)' for long-running operations",  # noqa: E501
            # "- Use 'await ask_user(message, response_type)' to get user input during execution",  # noqa: E501
            # "  - response_type can be: 'string', 'bool', 'int', 'float', 'json'",
            "- DO NOT call asyncio.run() or try to run the main function yourself",
            "- DO NOT import asyncio or other modules - tools are already available",
            "- Example:",
            "    async def main():",
            "        result = await open(url='https://example.com', new=2)",
            "        return result",
        ])

        return "\n".join(parts)

    def generate_execution_namespace(self) -> dict[str, Any]:
        """Build Python namespace with tool functions and generated models."""
        namespace = {
            "__builtins__": __builtins__,
            "_result": None,
        }

        # Add tool functions
        for tool in self.tools:

            def make_tool_func(t: Tool):
                async def tool_func(*args, **kwargs):
                    try:
                        result = await t.execute(*args, **kwargs)
                        # Handle coroutines that weren't properly awaited
                        if inspect.iscoroutine(result):
                            result = await result
                        # Ensure we return a serializable value

                    except Exception as e:  # noqa: BLE001
                        return f"Error executing {t.name}: {e!s}"
                    else:
                        return (
                            result
                            if result is not None
                            else "Operation completed successfully"
                        )

                tool_func.__name__ = t.name
                tool_func.__doc__ = t.description
                return tool_func

            namespace[tool.name] = make_tool_func(tool)

        # Add generated model classes to namespace
        models_code = self.generate_return_models()
        if models_code:
            with contextlib.suppress(Exception):
                exec(models_code, namespace)

        return namespace

    def generate_return_models(self) -> str:
        """Generate Pydantic models for tool return types."""
        model_parts = []

        for tool in self.tools:
            generator = ToolCodeGenerator.from_tool(tool)
            if code := generator.generate_return_model():
                model_parts.append(code)

        return "\n\n".join(model_parts) if model_parts else ""
