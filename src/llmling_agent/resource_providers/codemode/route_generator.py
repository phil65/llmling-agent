"""Generate FastAPI routes for tools with automatic parameter parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from schemez.schema import json_schema_to_base_model


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent.tools.base import Tool


def generate_tool_routes(app: FastAPI, tools: list[Tool]) -> None:
    """Generate FastAPI routes for tools with automatic parameter parsing.

    Args:
        app: FastAPI application instance
        tools: List of tools to generate routes for
    """
    for tool in tools:
        _add_tool_route(app, tool)


def _add_tool_route(app: FastAPI, tool: Tool) -> None:
    """Add a single tool route to the FastAPI app.

    Args:
        app: FastAPI application instance
        tool: Tool to create route for
    """
    # Extract parameter information from tool schema
    schema = tool.schema["function"]
    parameters_schema = schema.get("parameters", {})

    # Create Pydantic model for parameter validation using schemez
    if parameters_schema.get("properties"):
        param_cls = json_schema_to_base_model(parameters_schema)  # type: ignore
    else:
        # Tool has no parameters
        param_cls = None

    # Create the route handler
    async def route_handler(*args, **kwargs) -> Any:
        """Route handler for the tool."""
        if param_cls:
            params_instance = param_cls(**kwargs)  # Parse and validate parameters
            dct = params_instance.model_dump()  # Convert to dict and remove None values
            clean_params = {k: v for k, v in dct.items() if v is not None}
            result = await _execute_tool(tool, **clean_params)
        else:
            result = await _execute_tool(tool)
        return {"result": result}

    # Set up the route with proper parameter annotations for FastAPI
    if param_cls:
        # Get field information from the generated model
        model_fields = param_cls.model_fields
        route_params = []
        for name, field_info in model_fields.items():
            field_type = field_info.annotation
            if field_info.is_required():
                route_params.append(f"{name}: {field_type.__name__}")  # type: ignore
            else:
                route_params.append(f"{name}: {field_type.__name__} = None")  # type: ignore

        # Create function signature dynamically
        param_str = ", ".join(route_params)
        func_code = f"""
async def dynamic_handler({param_str}) -> dict[str, Any]:
    kwargs = {{{", ".join(f'"{name}": {name}' for name in model_fields)}}}
    return await route_handler(**kwargs)
"""
        # Execute the dynamic function creation
        namespace = {"route_handler": route_handler, "Any": Any}
        exec(func_code, namespace)
        dynamic_handler: Callable = namespace["dynamic_handler"]  # type: ignore
    else:

        async def dynamic_handler() -> dict[str, Any]:
            return await route_handler()

    # Add route to FastAPI app
    app.get(f"/tools/{tool.name}")(dynamic_handler)


async def _execute_tool(tool: Tool, **kwargs) -> Any:
    """Execute a tool with the given parameters.

    Args:
        tool: Tool to execute
        **kwargs: Tool parameters

    Returns:
        Tool execution result
    """
    try:
        # For now, just simulate execution
        # In real implementation, this would call the actual tool
        # potentially through sandbox providers
        return f"Executed {tool.name} with params: {kwargs}"
    except Exception as e:  # noqa: BLE001
        return f"Error executing {tool.name}: {e!s}"


if __name__ == "__main__":
    from fastapi import FastAPI

    from llmling_agent.tools.base import Tool

    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    # Create FastAPI app
    app = FastAPI(title="Tool API")

    # Create tools
    tools: list[Tool] = [
        Tool.from_callable(add_numbers),
        Tool.from_callable(greet),
    ]

    # Generate routes
    generate_tool_routes(app, tools)

    # Print available routes
    print("Generated routes:")
    for route in app.routes:
        try:
            if route.path.startswith("/tools/"):  # type: ignore[attr-defined]
                print(f"  GET {route.path}")  # type: ignore[attr-defined]
        except AttributeError:
            continue

    # To run: uvicorn route_generator:app --reload
    print("\nTo test: uvicorn route_generator:app --reload")
    print("Then visit: http://localhost:8000/tools/add_numbers?x=5&y=3")
