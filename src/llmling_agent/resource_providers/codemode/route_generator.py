"""Generate FastAPI routes for tools with automatic parameter parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from pydantic import create_model


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
    params = schema.get("parameters", {}).get("properties", {})
    required = set(schema.get("parameters", {}).get("required", []))

    # Build Pydantic model fields
    model_fields: dict[str, tuple[type, Any]] = {}
    for name, param_info in params.items():
        python_type = _get_python_type(param_info)  # type: ignore[arg-type]
        if name in required:
            model_fields[name] = (python_type, ...)
        else:
            model_fields[name] = (python_type, None)

    # Create Pydantic model for parameter validation
    if model_fields:
        param_cls = create_model(f"{tool.name.title()}Params", **model_fields)  # type: ignore
    else:
        # Tool has no parameters
        param_cls = None

    # Create the route handler
    async def route_handler(*args, **kwargs) -> Any:
        """Route handler for the tool."""
        if param_cls:
            # Parse and validate parameters
            params_instance = param_cls(**kwargs)
            # Convert to dict and remove None values
            clean_params = {
                k: v for k, v in params_instance.model_dump().items() if v is not None
            }
            result = await _execute_tool(tool, **clean_params)
        else:
            result = await _execute_tool(tool)

        return {"result": result}

    # Set up the route with proper parameter annotations for FastAPI
    if param_cls:
        # Add query parameters to route handler signature
        route_params = []
        for name, (field_type, default) in model_fields.items():
            if default is ...:  # Required field
                route_params.append(f"{name}: {field_type.__name__}")
            else:
                route_params.append(f"{name}: {field_type.__name__} = None")

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


def _get_python_type(param_info: dict[str, Any]) -> type:
    """Convert JSON schema type to Python type.

    Args:
        param_info: Parameter schema information

    Returns:
        Python type for the parameter
    """
    param_type = param_info.get("type", "string")
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return type_mapping.get(param_type, str)


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
        if hasattr(route, "path") and route.path.startswith("/tools/"):
            print(f"  GET {route.path}")

    # To run: uvicorn route_generator:app --reload
    print("\nTo test: uvicorn route_generator:app --reload")
    print("Then visit: http://localhost:8000/tools/add_numbers?x=5&y=3")
