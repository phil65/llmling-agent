"""Helper functions for FastAPI route generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from schemez.schema import json_schema_to_base_model


if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel
    from pydantic.fields import FieldInfo


def create_param_model(parameters_schema: dict[str, Any]) -> type[BaseModel] | None:
    """Create Pydantic model for parameter validation using schemez.

    Args:
        parameters_schema: JSON schema for tool parameters

    Returns:
        Pydantic model class or None if no parameters
    """
    if parameters_schema.get("properties"):
        return json_schema_to_base_model(parameters_schema)  # type: ignore
    return None


def generate_func_code(model_fields: dict[str, FieldInfo]) -> str:
    """Generate dynamic function code for FastAPI route handler.

    Args:
        model_fields: Model fields from Pydantic model

    Returns:
        Generated function code as string
    """
    route_params = []
    for name, field_info in model_fields.items():
        field_type = field_info.annotation
        if field_info.is_required():
            route_params.append(f"{name}: {field_type.__name__}")  # type: ignore
        else:
            route_params.append(f"{name}: {field_type.__name__} = None")  # type: ignore

    # Create function signature dynamically
    param_str = ", ".join(route_params)
    return f"""
async def dynamic_handler({param_str}) -> dict[str, Any]:
    kwargs = {{{", ".join(f'"{name}": {name}' for name in model_fields)}}}
    return await route_handler(**kwargs)
"""


def create_route_handler(tool_callable: Callable, param_cls: type | None) -> Callable:
    """Create FastAPI route handler for a tool.

    Args:
        tool_callable: The tool function to execute
        param_cls: Pydantic model for parameter validation

    Returns:
        Async route handler function
    """

    async def route_handler(*args, **kwargs) -> Any:
        """Route handler for the tool."""
        if param_cls:
            params_instance = param_cls(**kwargs)  # Parse and validate parameters
            dct = params_instance.model_dump()  # Convert to dict and remove None values
            clean_params = {k: v for k, v in dct.items() if v is not None}
            result = await _execute_tool_function(tool_callable, **clean_params)
        else:
            result = await _execute_tool_function(tool_callable)
        return {"result": result}

    return route_handler


async def _execute_tool_function(tool_callable: Callable, **kwargs) -> Any:
    """Execute a tool function with the given parameters.

    Args:
        tool_callable: Tool function to execute
        **kwargs: Tool parameters

    Returns:
        Tool execution result
    """
    try:
        # For now, just simulate execution
        # In real implementation, this would call the actual tool
        # potentially through sandbox providers
        return f"Executed {tool_callable.__name__} with params: {kwargs}"
    except Exception as e:  # noqa: BLE001
        return f"Error executing {tool_callable.__name__}: {e!s}"


if __name__ == "__main__":
    from llmling_agent.tools.base import Tool

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    # Create a tool and demonstrate helper functions
    tool = Tool.from_callable(greet)
    schema = tool.schema["function"]
    parameters_schema = schema.get("parameters", {})

    # Create parameter model
    param_cls = create_param_model(dict(parameters_schema))
    print(f"Generated parameter model: {param_cls}")

    if param_cls:
        print(f"Model fields: {param_cls.model_fields}")

        # Generate function code
        func_code = generate_func_code(param_cls.model_fields)
        print(f"Generated function code:\n{func_code}")

    # Create route handler
    handler = create_route_handler(greet, param_cls)
    print(f"Generated route handler: {handler}")
