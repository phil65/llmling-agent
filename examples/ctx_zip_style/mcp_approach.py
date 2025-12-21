"""Demo of ctx-zip style code generation approach.

This example shows how to use code generation instead of HTTP servers
to make tools available in sandbox environments, especially useful for
cloud sandboxes like E2B that can't reach localhost.
"""

import asyncio

from exxec.configs import LocalExecutionEnvironmentConfig

from agentpool.resource_providers.codemode.remote_mcp_execution import (
    RemoteMCPExecutor,
)
from agentpool.tools.base import Tool


def add_numbers(x: int, y: int) -> int:
    """Add two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        Product of x and y
    """
    return x * y


async def fetch_weather(city: str, country: str = "US") -> dict:
    """Fetch weather information for a city.

    Args:
        city: Name of the city
        country: Country code (default: US)

    Returns:
        Weather information dict
    """
    return {"city": city, "country": country, "temperature": 22}


async def demo_code_generation_approach():
    """Demo new code generation approach (ctx-zip style, works with cloud sandboxes)."""
    print("\n=== Code Generation Approach (ctx-zip style) ===")
    tools = [
        Tool.from_callable(add_numbers),
        Tool.from_callable(multiply_numbers),
        Tool.from_callable(fetch_weather),
    ]
    config = LocalExecutionEnvironmentConfig()  # Could be E2B, Modal, etc.

    provider = RemoteMCPExecutor.from_tools(tools, config)

    async with provider:
        print("Tool description:")
        print(provider.get_tool_description())
        print("\nExecuting code with direct imports...")

        # Code that imports tools directly (no HTTP calls)
        code = """
from tools.add_numbers import add_numbers
from tools.multiply_numbers import multiply_numbers
from tools.fetch_weather import fetch_weather

# Direct function calls - no HTTP server needed!
result1 = add_numbers(15, 27)
result2 = multiply_numbers(4, 7)
result3 = await fetch_weather("San Francisco", "US")

_result = {
    "addition": result1,
    "multiplication": result2,
    "weather": result3
}
print(f"Addition: {result1}")
print(f"Multiplication: {result2}")
print(f"Weather: {result3}")
"""
        result = await provider.execute_code(code)
        print(f"Final result: {result.result}")


async def demo_inspect_generated_files():
    """Show what files are generated in the sandbox."""
    print("\n=== Inspecting Generated Files ===")
    tools = [Tool.from_callable(add_numbers), Tool.from_callable(multiply_numbers)]
    config = LocalExecutionEnvironmentConfig()

    provider = RemoteMCPExecutor.from_tools(tools, config)

    async with provider:
        # List generated files
        list_code = """
import os
print("Generated files in sandbox:")
for root, dirs, files in os.walk("tools"):
    for file in files:
        filepath = os.path.join(root, file)
        print(f"  {filepath}")

# Show content of a generated tool file
print("\\nContent of tools/add_numbers.py:")
with open("tools/add_numbers.py", "r") as f:
    content = f.read()
    print(content)
"""
        await provider.execute_code(list_code)


async def main():
    """Run all demos."""
    await demo_code_generation_approach()
    await demo_inspect_generated_files()


if __name__ == "__main__":
    asyncio.run(main())
