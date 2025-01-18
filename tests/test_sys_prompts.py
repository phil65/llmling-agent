from __future__ import annotations

from llmling import LLMCallableTool
from pydantic import BaseModel
import pytest

from llmling_agent import Agent


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello {name}!"


@pytest.fixture
def agent():
    """Create a test agent with some tools."""
    agent = Agent[None](
        name="tester",
        description="A test agent",
        system_prompt=[
            "Be helpful",
            "Be concise",
        ],
    )
    # Add real tools
    agent.tools.register_tool(LLMCallableTool.from_callable(add))
    agent.tools.register_tool(LLMCallableTool.from_callable(greet))
    return agent


class TestPersonality(BaseModel):
    role: str = "I am a test agent"
    style: str = "I speak formally"


@pytest.mark.asyncio
async def test_basic_prompt(agent):
    """Test basic prompt without special features."""
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "You are tester" in result
    assert "A test agent" in result
    assert "Be helpful" in result
    assert "Be concise" in result
    assert "add" not in result  # Tools not included by default


@pytest.mark.asyncio
async def test_tool_injection(agent):
    """Test tool injection in different modes."""
    # Enable tool injection
    agent.sys_prompts.inject_tools = "all"
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "You have access to these tools:" in result
    assert "add: Add two numbers" in result
    assert "greet: Greet someone" in result

    # Change to strict mode
    agent.sys_prompts.tool_usage_style = "strict"
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "You MUST use these tools" in result
    assert "Do not attempt to perform tasks without using appropriate tools" in result


@pytest.mark.asyncio
async def test_agent_info_control(agent):
    """Test control of agent info injection."""
    # Default includes agent info
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "You are tester" in result
    assert "A test agent" in result

    # Disable agent info
    agent.sys_prompts.inject_agent_info = False
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "You are tester" not in result
    assert "A test agent" not in result
    assert "Be helpful" in result  # Other prompts still included


@pytest.mark.asyncio
async def test_structured_prompt():
    """Test using a pydantic model as prompt."""
    agent = Agent[None](name="structured", system_prompt=TestPersonality())
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "I am a test agent" in result
    assert "I speak formally" in result


@pytest.mark.asyncio
async def test_custom_template(agent):
    """Test using a custom template."""
    agent.sys_prompts.template = """
    Agent: {{ agent.name }}
    {% for p in prompts %}
    - {{ p }}
    {% endfor %}
    """
    result = await agent.sys_prompts.format_system_prompt(agent)
    assert "Agent: tester" in result
    assert "- Be helpful" in result
    assert "- Be concise" in result


@pytest.mark.asyncio
async def test_dynamic_evaluation(agent):
    """Test dynamic vs cached prompt evaluation."""
    counter = 0

    async def dynamic_prompt():
        nonlocal counter
        counter += 1
        return f"Count: {counter}"

    agent.sys_prompts.prompts.append(dynamic_prompt)

    # Dynamic (default)
    result1 = await agent.sys_prompts.format_system_prompt(agent)
    result2 = await agent.sys_prompts.format_system_prompt(agent)
    assert "Count: 1" in result1
    assert "Count: 2" in result2

    # Cached
    agent.sys_prompts.dynamic = False
    await agent.sys_prompts.refresh_cache()  # Counter becomes 3
    result3 = await agent.sys_prompts.format_system_prompt(agent)
    result4 = await agent.sys_prompts.format_system_prompt(agent)
    assert "Count: 3" in result3
    assert "Count: 3" in result4  # Same as before due to caching
