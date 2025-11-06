"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

import contextlib
import inspect
from typing import TYPE_CHECKING, Any

from schemez import create_schema

from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Sequence

    from schemez.typedefs import Property


TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "null": "None",
}


def _extract_basic_signature(tool: Tool, return_type: str = "Any") -> str:
    """Fallback signature extraction from tool schema."""
    schema = tool.schema["function"]
    params = schema.get("parameters", {}).get("properties", {})
    required = set(schema.get("required", []))  # type: ignore

    param_strs = []
    for name, param_info in params.items():
        # Use improved type inference
        type_hint = _infer_parameter_type(tool, name, param_info)

        if name not in required:
            param_strs.append(f"{name}: {type_hint} = None")
        else:
            param_strs.append(f"{name}: {type_hint}")

    return f"{tool.name}({', '.join(param_strs)}) -> {return_type}"


def _infer_parameter_type(tool: Tool, param_name: str, param_info: Property) -> str:
    """Infer parameter type from schema and function inspection."""
    schema_type = param_info.get("type", "Any")

    # If schema has a specific type, use it
    if schema_type != "object":
        return TYPE_MAP.get(schema_type, "Any")

    # For 'object' type, try to infer from function signature
    try:
        callable_func = tool.callable
        sig = inspect.signature(callable_func)

        if param_name in sig.parameters:
            param = sig.parameters[param_name]

            # Try annotation first
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__name__"):
                    return param.annotation.__name__
                return str(param.annotation)

            # Infer from default value
            if param.default != inspect.Parameter.empty:
                default_type = type(param.default).__name__
                # Map common types
                if default_type in ["int", "float", "str", "bool"]:
                    return default_type
            # If no default and it's required, assume str for web-like functions
            required = set(
                tool.schema.get("function", {}).get("parameters", {}).get("required", [])
            )
            if param_name in required:
                return "str"

    except Exception:  # noqa: BLE001
        pass

    # Fallback to Any for unresolved object types
    return "Any"


def _get_return_model_name(tool: Tool) -> str:
    """Get the return model name for a tool."""
    try:
        callable_func = tool.callable
        schema = create_schema(callable_func)
        return_schema = schema.returns

        if return_schema.get("type") == "object":
            return f"{tool.name.title()}Response"
        if return_schema.get("type") == "array":
            return f"list[{tool.name.title()}Item]"
        return TYPE_MAP.get(return_schema.get("type", "string"), "Any")
    except Exception:  # noqa: BLE001
        return "Any"


async def _get_function_signature(tool: Tool) -> str:
    """Extract function signature using schemez."""
    try:
        return_model_name = _get_return_model_name(tool)
        return _extract_basic_signature(tool, return_model_name)
    except Exception:  # noqa: BLE001
        return _extract_basic_signature(tool, "Any")


def _generate_return_models(all_tools: list[Tool]) -> str:
    """Generate Pydantic models for tool return types using schemez."""
    model_parts = []

    for tool in all_tools:
        try:
            callable_func = tool.callable
            schema = create_schema(callable_func)

            if schema.returns.get("type") not in {"object", "array"}:
                continue

            class_name = f"{tool.name.title()}Response"
            model_code = schema.to_pydantic_model_code(class_name=class_name)

            if model_code.strip():
                model_parts.append(model_code.strip())

        except Exception:  # noqa: BLE001
            continue

    return "\n\n".join(model_parts) if model_parts else ""


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
                signature = await _get_function_signature(tool)
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
