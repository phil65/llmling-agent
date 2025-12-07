"""Tests for the observability providers."""

import os

import pytest

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent.observability import registry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the global registry state before each test."""
    # Reset the configured state
    registry._configured = False
    yield
    # Clean up after test
    registry._configured = False


LOGFIRE_MANIFEST = """
observability:
  enabled: true
  provider:
    type: logfire
    token: {logfire_token}
    service_name: integration_test
    environment: test

agents:
  test_agent:
    name: test_agent
    model:
      type: test
      custom_output_text: "Test response"
    system_prompts:
      - You are a helpful assistant.
"""

LANGSMITH_MANIFEST = """
observability:
  enabled: true
  provider:
    type: langsmith
    api_key: {langsmith_key}
    project_name: integration_test
    service_name: integration_test
    environment: test

agents:
  test_agent:
    name: test_agent
    model:
      type: test
      custom_output_text: "Test response"
    system_prompts:
      - You are a helpful assistant.
"""

CUSTOM_MANIFEST = """
observability:
  enabled: true
  provider:
    type: custom
    endpoint: http://localhost:4318
    headers:
      Authorization: Bearer test_token
    service_name: integration_test
    environment: test

agents:
  test_agent:
    name: test_agent
    model:
      type: test
      custom_output_text: "Test response"
    system_prompts:
      - You are a helpful assistant.
"""


@pytest.mark.skipif(
    not os.getenv("LOGFIRE_TOKEN"),
    reason="LOGFIRE_TOKEN not configured",
)
async def test_logfire_provider_integration():
    """Test that Logfire provider can be initialized and works."""
    manifest_str = LOGFIRE_MANIFEST.format(logfire_token=os.getenv("LOGFIRE_TOKEN", "dummy_token"))

    manifest = AgentsManifest.from_yaml(manifest_str)

    async with AgentPool(manifest) as pool:
        agent = pool.get_agent("test_agent")
        # Run a simple prompt
        result = await agent.run("Hello!")
        # Verify no errors occurred
        assert result.content == "Test response"
        assert registry._configured


@pytest.mark.skipif(
    not os.getenv("LANGSMITH_API_KEY"),
    reason="LANGSMITH_API_KEY not configured",
)
async def test_langsmith_provider_integration():
    """Test that Langsmith provider can be initialized and works."""
    manifest_str = LANGSMITH_MANIFEST.format(
        langsmith_key=os.getenv("LANGSMITH_API_KEY", "dummy_key")
    )

    manifest = AgentsManifest.from_yaml(manifest_str)

    async with AgentPool(manifest) as pool:
        agent = pool.get_agent("test_agent")
        # Run a simple prompt
        result = await agent.run("Hello!")
        # Verify no errors occurred
        assert result.content == "Test response"
        assert registry._configured


async def test_custom_provider_integration():
    """Test that custom provider configuration works."""
    manifest_str = CUSTOM_MANIFEST

    manifest = AgentsManifest.from_yaml(manifest_str)

    async with AgentPool(manifest) as pool:
        agent = pool.get_agent("test_agent")
        # Run a simple prompt
        result = await agent.run("Hello!")
        # Verify no errors occurred
        assert result.content == "Test response"
        assert registry._configured


async def test_disabled_observability():
    """Test that disabled observability works without configuration."""
    manifest_str = """
observability:
  enabled: false

agents:
  test_agent:
    name: test_agent
    model:
      type: test
      custom_output_text: "Test response"
    system_prompts:
      - You are a helpful assistant.
"""

    manifest = AgentsManifest.from_yaml(manifest_str)

    async with AgentPool(manifest) as pool:
        agent = pool.get_agent("test_agent")
        # Run a simple prompt
        result = await agent.run("Hello!")
        # Verify no errors occurred
        assert result.content == "Test response"
        # Registry should not be configured when disabled
        assert not registry._configured


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
