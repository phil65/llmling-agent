"""Test configuration and shared fixtures."""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai.models.test import TestModel
import pytest
import yamling

from llmling_agent import Agent, AgentConfig, AgentPool, AgentsManifest


TEST_RESPONSE = "I am a test response"


@pytest.fixture
def default_model() -> str:
    """Default model for testing."""
    return "openai:gpt-5-nano"


@pytest.fixture
def vision_model() -> str:
    """Vision-capable model for testing."""
    return "openai:gpt-5-nano"


@pytest.fixture(scope="session", autouse=True)
def disable_logfire():
    """Disable logfire for all tests."""
    # Set environment variable to disable logfire
    os.environ["LOGFIRE_DISABLE"] = "true"
    # Also disable observability entirely
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    # Mock logfire configure to be a no-op
    try:
        import logfire

        original_configure = logfire.configure
        logfire.configure = lambda *args, **kwargs: None  # type: ignore
        yield
        logfire.configure = original_configure
    except ImportError:
        # logfire not available, nothing to disable
        yield


VALID_CONFIG = """\
responses:
  SupportResult:
    response_schema:
        type: inline
        description: Support agent response
        fields:
            advice:
                type: str
                description: Support advice
            risk:
                type: int
                ge: 0
                le: 100
  ResearchResult:
    response_schema:
        type: inline
        description: Research agent response
        fields:
            findings:
                type: str
                description: Research findings

agents:
  support:
    display_name: Support Agent
    model: {default_model}
    output_type: SupportResult
    system_prompts:
      - You are a support agent
      - "Context: {{data}}"
  researcher:
    display_name: Research Agent
    model: {default_model}
    output_type: ResearchResult
    system_prompts:
      - You are a researcher
"""


@pytest.fixture
def valid_config(default_model: str) -> dict[str, Any]:
    """Fixture providing valid agent configuration."""
    return yamling.load_yaml(VALID_CONFIG.format(default_model=default_model), verify_type=dict)


@pytest.fixture
def test_agent() -> Agent[None]:
    """Create an agent with TestModel for testing."""
    model = TestModel(custom_output_text=TEST_RESPONSE)
    return Agent(name="test-agent", model=model)


@pytest.fixture
def manifest():
    """Create test manifest with some agents."""
    agent_1 = AgentConfig(name="agent1", model="test")
    agent_2 = AgentConfig(name="agent2", model="test")
    return AgentsManifest(agents={"agent1": agent_1, "agent2": agent_2})


@pytest.fixture
async def pool(manifest):
    """Create test pool with agents."""
    async with AgentPool(manifest) as pool:
        yield pool
