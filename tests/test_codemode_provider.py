"""Integration test for codemode providers."""

from anyenv.code_execution.configs import LocalExecutionEnvironmentConfig
import pytest

from llmling_agent import Agent
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.resource_providers.codemode.secure_provider import (
    SecureCodeModeResourceProvider,
)
from llmling_agent.tools.base import Tool


def add_numbers(x: int, y: int) -> int:
    """Add two numbers together.

    Args:
        x: First number
        y: Second number
    """
    return x + y


async def fetch_data(name: str, count: int = 1) -> dict:
    """Fetch some mock data.

    Args:
        name: Name to use in response
        count: Number of items to return
    """
    return {"name": name, "count": count, "items": list(range(count))}


@pytest.mark.asyncio
async def test_secure_codemode_provider_creation():
    """Test that SecureCodeModeResourceProvider can be created without sig errors."""
    tools = [
        Tool.from_callable(add_numbers),
        Tool.from_callable(fetch_data),
    ]

    config = LocalExecutionEnvironmentConfig()
    provider = SecureCodeModeResourceProvider(
        providers=[StaticResourceProvider(tools=tools)], execution_config=config
    )

    provider_tools = await provider.get_tools()
    assert len(provider_tools) == 1  # Single execute tool
    execute_tool = provider_tools[0]
    assert execute_tool.name.startswith("execute")  # Could be execute_tool or execute
    assert "Add two numbers" in execute_tool.description
    assert "Fetch some mock data" in execute_tool.description


@pytest.mark.asyncio
async def test_secure_codemode_provider_with_agent():
    """Test that SecureCodeModeResourceProvider works with Agent without errors."""
    tools = [
        Tool.from_callable(add_numbers),
        Tool.from_callable(fetch_data),
    ]

    config = LocalExecutionEnvironmentConfig()
    provider = SecureCodeModeResourceProvider(
        providers=[StaticResourceProvider(tools=tools)], execution_config=config
    )

    agent = Agent(model="test", toolsets=[provider])
    async with agent:
        # Verify tool is properly registered
        available_tools = await agent.tools.list_tools()
        assert len(available_tools) == 1
        tool_names = list(available_tools.keys())
        assert any(name.startswith("execute") for name in tool_names)
        assert all(available_tools.values())


@pytest.mark.asyncio
async def test_codemode_tool_schema_generation():
    """Test that tool schema generation works properly for codemode providers."""
    tools = [Tool.from_callable(add_numbers)]

    config = LocalExecutionEnvironmentConfig()
    provider = SecureCodeModeResourceProvider(
        providers=[StaticResourceProvider(tools=tools)], execution_config=config
    )

    provider_tools = await provider.get_tools()
    execute_tool = provider_tools[0]

    # Should be able to get schema without errors
    schema = execute_tool.schema
    assert schema["type"] == "function"
    assert "function" in schema

    function_def = schema["function"]
    assert "name" in function_def
    assert "description" in function_def
    assert "parameters" in function_def

    # Should have python_code parameter
    params = function_def["parameters"]
    assert "properties" in params
    assert "python_code" in params["properties"]


@pytest.mark.asyncio
async def test_multiple_providers():
    """Test SecureCodeModeResourceProvider with multiple underlying providers."""
    tools1 = [Tool.from_callable(add_numbers)]
    tools2 = [Tool.from_callable(fetch_data)]

    provider1 = StaticResourceProvider(tools=tools1)
    provider2 = StaticResourceProvider(tools=tools2)

    config = LocalExecutionEnvironmentConfig()
    secure_provider = SecureCodeModeResourceProvider(
        providers=[provider1, provider2], execution_config=config
    )

    # Should aggregate tools from all providers
    provider_tools = await secure_provider.get_tools()
    assert len(provider_tools) == 1

    execute_tool = provider_tools[0]
    # Description should include info about all tools
    assert "add_numbers" in execute_tool.description
    assert "fetch_data" in execute_tool.description
