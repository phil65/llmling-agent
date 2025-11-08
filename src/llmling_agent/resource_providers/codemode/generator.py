"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING

from schemez import create_schema

from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Callable

    from schemez.typedefs import OpenAIFunctionTool, Property


TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "null": "None",
}


@dataclass
class CodeGenerator:
    """Meta-resource provider that exposes tools through Python execution."""

    schema: OpenAIFunctionTool
    """Schema of the tool."""

    callable: Callable
    """Tool to generate code for."""

    name_override: str | None = None
    """Name of the tool."""

    @classmethod
    def from_tool(cls, tool: Tool) -> CodeGenerator:
        """Create a CodeGenerator from a Tool."""
        return cls(schema=tool.schema, callable=tool.callable, name_override=tool.name)

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.name_override or self.callable.__name__

    def _extract_basic_signature(self, return_type: str = "Any") -> str:
        """Fallback signature extraction from tool schema."""
        schema = self.schema["function"]
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("required", []))  # type: ignore

        param_strs = []
        for name, param_info in params.items():
            # Use improved type inference
            type_hint = self._infer_parameter_type(name, param_info)

            if name not in required:
                param_strs.append(f"{name}: {type_hint} = None")
            else:
                param_strs.append(f"{name}: {type_hint}")

        return f"{self.name}({', '.join(param_strs)}) -> {return_type}"

    def _infer_parameter_type(self, param_name: str, param_info: Property) -> str:
        """Infer parameter type from schema and function inspection."""
        schema_type = param_info.get("type", "Any")

        # If schema has a specific type, use it
        if schema_type != "object":
            return TYPE_MAP.get(schema_type, "Any")

        # For 'object' type, try to infer from function signature
        try:
            callable_func = self.callable
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
                    self.schema.get("function", {})
                    .get("parameters", {})
                    .get("required", [])
                )
                if param_name in required:
                    return "str"

        except Exception:  # noqa: BLE001
            pass

        # Fallback to Any for unresolved object types
        return "Any"

    def _get_return_model_name(self) -> str:
        """Get the return model name for a tool."""
        try:
            callable_func = self.callable
            schema = create_schema(callable_func)
            return_schema = schema.returns

            if return_schema.get("type") == "object":
                return f"{self.name.title()}Response"
            if return_schema.get("type") == "array":
                return f"list[{self.name.title()}Item]"
            return TYPE_MAP.get(return_schema.get("type", "string"), "Any")
        except Exception:  # noqa: BLE001
            return "Any"

    def get_function_signature(self) -> str:
        """Extract function signature using schemez."""
        try:
            return_model_name = self._get_return_model_name()
            return self._extract_basic_signature(return_model_name)
        except Exception:  # noqa: BLE001
            return self._extract_basic_signature("Any")


if __name__ == "__main__":
    import webbrowser

    tool = Tool.from_callable(webbrowser.open)
    generator = CodeGenerator.from_tool(tool)
    sig = generator.get_function_signature()
    print(sig)
