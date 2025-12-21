"""Demo of ctx-zip style code generation approach.

This example shows how to use code generation instead of HTTP servers
to make tools available in sandbox environments, especially useful for
cloud sandboxes like E2B that can't reach localhost.
"""

import asyncio

from exxec.configs import LocalExecutionEnvironmentConfig

from agentpool import Agent
from agentpool.resource_providers import StaticResourceProvider
from agentpool.resource_providers.codemode import RemoteCodeModeResourceProvider
from agentpool.tools.base import Tool


def add_numbers(x: int, y: int) -> int:
    """Add two numbers together.

    Args:
        x: First number
        y: Second number
    """
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together.

    Args:
        x: First number
        y: Second number
    """
    return x * y


async def fetch_weather(city: str, country: str = "US") -> dict:
    """Fetch weather information for a city.

    Args:
        city: Name of the city
        country: Country code (default: US)
    """
    return {"city": city, "country": country, "temperature": 22}


async def demo_http_server_approach():
    """Demo original HTTP server approach (works with local/docker)."""
    print("=== HTTP Server Approach ===")
    tools = [
        Tool.from_callable(add_numbers),
        Tool.from_callable(multiply_numbers),
        Tool.from_callable(fetch_weather),
    ]

    config = LocalExecutionEnvironmentConfig()
    toolset = RemoteCodeModeResourceProvider(
        providers=[StaticResourceProvider(tools=tools)], execution_config=config
    )
    agent = Agent(model="openai:gpt-5-nano", toolsets=[toolset])
    async with agent:
        result = await agent.run(
            "What code is available for your execute tool? Which calls can you make?"
        )
        print(result.format())


if __name__ == "__main__":
    asyncio.run(demo_http_server_approach())
