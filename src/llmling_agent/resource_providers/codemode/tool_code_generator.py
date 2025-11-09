"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from schemez import create_schema

from llmling_agent.resource_providers.codemode.route_helpers import (
    create_param_model,
    create_route_handler,
    generate_func_code,
)
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import FastAPI
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
class ToolCodeGenerator:
    """Generates code artifacts for a single tool."""

    schema: OpenAIFunctionTool
    """Schema of the tool."""

    callable: Callable
    """Tool to generate code for."""

    name_override: str | None = None
    """Name of the tool."""

    @classmethod
    def from_tool(cls, tool: Tool) -> ToolCodeGenerator:
        """Create a ToolCodeGenerator from a Tool."""
        return cls(schema=tool.schema, callable=tool.callable, name_override=tool.name)

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.name_override or self.callable.__name__

    def _extract_basic_signature(self, return_type: str = "Any") -> str:
        """Fallback signature extraction from tool schema."""
        schema = self.schema["function"]
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("parameters", {}).get("required", []))

        param_strs = []
        for name, param_info in params.items():
            # Skip context parameters that should be hidden from users
            if self._is_context_parameter(name):
                continue

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
            # Use wrapped signature if available (for context parameter hiding)
            sig = getattr(callable_func, "__signature__", None) or inspect.signature(
                callable_func
            )

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
            schema = create_schema(self.callable)
            if schema.returns.get("type") == "object":
                return f"{self.name.title()}Response"
            if schema.returns.get("type") == "array":
                return f"list[{self.name.title()}Item]"
            return TYPE_MAP.get(schema.returns.get("type", "string"), "Any")
        except Exception:  # noqa: BLE001
            return "Any"

    def get_function_signature(self) -> str:
        """Extract function signature using schemez."""
        try:
            return_model_name = self._get_return_model_name()
            return self._extract_basic_signature(return_model_name)
        except Exception:  # noqa: BLE001
            return self._extract_basic_signature("Any")

    def _get_callable_signature(self) -> inspect.Signature:
        """Get signature from callable, respecting wrapped signatures."""
        # Use wrapped signature if available (for context parameter hiding)
        return getattr(self.callable, "__signature__", None) or inspect.signature(
            self.callable
        )

    def _is_context_parameter(self, param_name: str) -> bool:  # noqa: PLR0911
        """Check if a parameter is a context parameter that should be hidden."""
        try:
            sig = self._get_callable_signature()
            if param_name not in sig.parameters:
                return False

            param = sig.parameters[param_name]
            if param.annotation == inspect.Parameter.empty:
                return False

            # Check if parameter is RunContext or AgentContext
            annotation = param.annotation
            annotation_str = str(annotation)

            # Handle RunContext (including parameterized like RunContext[None])
            if annotation is RunContext:
                return True

            # Check for parameterized RunContext using string matching
            if "RunContext" in annotation_str:
                return True

            # Handle AgentContext
            if hasattr(annotation, "__name__") and annotation.__name__ == "AgentContext":
                return True

            # Check for AgentContext in string representation
            if "AgentContext" in annotation_str:
                return True

        except Exception:  # noqa: BLE001
            pass

        return False

    def generate_return_model(self) -> str | None:
        """Generate Pydantic model code for the tool's return type."""
        try:
            schema = create_schema(self.callable)
            if schema.returns.get("type") not in {"object", "array"}:
                return None

            class_name = f"{self.name.title()}Response"
            model_code = schema.to_pydantic_model_code(class_name=class_name)
            return model_code.strip() or None

        except Exception:  # noqa: BLE001
            return None

    # Route generation methods
    def generate_route_handler(self) -> Callable:
        """Generate FastAPI route handler for this tool.

        Returns:
            Async route handler function
        """
        # Extract parameter schema
        schema = self.schema["function"]
        parameters_schema = schema.get("parameters", {})

        # Create parameter model
        param_cls = create_param_model(dict(parameters_schema))

        # Create route handler
        return create_route_handler(self.callable, param_cls)

    def add_route_to_app(self, app: FastAPI, path_prefix: str = "/tools") -> None:
        """Add this tool's route to FastAPI app.

        Args:
            app: FastAPI application instance
            path_prefix: Path prefix for the route
        """
        # Extract parameter schema
        schema = self.schema["function"]
        parameters_schema = schema.get("parameters", {})

        # Create parameter model
        param_cls = create_param_model(dict(parameters_schema))

        # Create the route handler
        route_handler = self.generate_route_handler()

        # Set up the route with proper parameter annotations for FastAPI
        if param_cls:
            # Get field information from the generated model
            model_fields = param_cls.model_fields
            func_code = generate_func_code(model_fields)
            # Execute the dynamic function creation
            namespace = {"route_handler": route_handler, "Any": Any}
            exec(func_code, namespace)
            dynamic_handler: Callable = namespace["dynamic_handler"]  # type: ignore
        else:

            async def dynamic_handler() -> dict[str, Any]:
                return await route_handler()

        # Add route to FastAPI app
        app.get(f"{path_prefix}/{self.name}")(dynamic_handler)


if __name__ == "__main__":
    import webbrowser

    t = Tool.from_callable(webbrowser.open)
    generator = ToolCodeGenerator.from_tool(t)
    sig = generator.get_function_signature()
    print(sig)
