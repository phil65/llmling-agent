"""Generates HTTP client code for tools."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.tools.base import Tool


class APIClientCodeGenerator:
    """Generates HTTP client code that makes GET requests to tool endpoints."""

    def __init__(self, base_url: str):
        """Initialize the generator.

        Args:
            base_url: Base URL for the API server
        """
        self.base_url = base_url.rstrip("/")

    def generate_client_code(self, tools: Sequence[Tool]) -> str:
        """Generate complete client code with all tool functions.

        Args:
            tools: Tools to generate client code for

        Returns:
            Complete Python code with async functions for each tool
        """
        parts = [
            "import httpx",
            "from typing import Any",
            "",
        ]

        for tool in tools:
            parts.append(self._generate_tool_function(tool))
            parts.append("")

        return "\n".join(parts)

    def _generate_tool_function(self, tool: Tool) -> str:
        """Generate HTTP client function for a single tool.

        Args:
            tool: Tool to generate function for

        Returns:
            Python function code as string
        """
        # Extract parameters from tool schema
        schema = tool.schema["function"]
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("parameters", {}).get("required", []))

        # Build function signature
        param_strs = []
        for name, param_info in params.items():
            type_hint = self._get_type_hint(param_info)  # type: ignore[arg-type]
            if name in required:
                param_strs.append(f"{name}: {type_hint}")
            else:
                param_strs.append(f"{name}: {type_hint} = None")

        signature = f"async def {tool.name}({', '.join(param_strs)}) -> Any:"

        # Build function body
        docstring = f'    """{tool.callable.__doc__ or ""}"""'

        # Build params dict, excluding None values
        param_assignments = [f"        '{name}': {name}," for name in params]
        params_dict = "    params = {\n" + "\n".join(param_assignments) + "\n    }"
        params_dict += "\n    # Remove None values\n    params = {k: v for k, v in params.items() if v is not None}"  # noqa: E501

        # Build HTTP request
        url = f"{self.base_url}/tools/{tool.name}"
        http_request = f"""    async with httpx.AsyncClient() as client:
        response = await client.get("{url}", params=params)
        response.raise_for_status()
        return response.json()"""

        return f"{signature}\n{docstring}\n{params_dict}\n{http_request}"

    def _get_type_hint(self, param_info: dict) -> str:
        """Get Python type hint from parameter schema.

        Args:
            param_info: Parameter schema information

        Returns:
            Python type hint as string
        """
        param_type = param_info.get("type", "Any")
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }
        return type_mapping.get(param_type, "Any")
