"""Test configuration and shared fixtures."""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai.models.test import TestModel
import pytest
import yamling

from agentpool import Agent, AgentPool, AgentsManifest, NativeAgentConfig


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
def disable_logfire(tmp_path_factory):
    """Disable logfire for all tests and set up test directories."""
    from pathlib import Path

    # Set environment variable to disable logfire
    os.environ["LOGFIRE_DISABLE"] = "true"
    # Also disable observability entirely
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    # Skip config dir override in CI - not needed and credentials aren't available anyway
    is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
    if not is_ci:
        # Use temp directory for Claude storage during tests
        claude_test_dir = tmp_path_factory.mktemp("claude_config")
        # Copy credentials file if it exists so integration tests can authenticate
        # Use copy instead of symlink for cross-platform compatibility (Windows needs admin/dev
        # mode for symlinks)
        real_creds = Path.home() / ".claude" / ".credentials.json"
        if real_creds.exists():
            import shutil

            test_creds = claude_test_dir / ".credentials.json"
            shutil.copy2(real_creds, test_creds)
        os.environ["CLAUDE_CONFIG_DIR"] = str(claude_test_dir)
        # Use temp directory for Codex data during tests
        codex_test_dir = tmp_path_factory.mktemp("codex_home")
        # Copy Codex auth file if it exists so integration tests can authenticate
        real_codex_auth = Path.home() / ".codex" / "auth.json"
        if real_codex_auth.exists():
            import shutil

            test_codex_auth = codex_test_dir / "auth.json"
            shutil.copy2(real_codex_auth, test_codex_auth)
        os.environ["CODEX_HOME"] = str(codex_test_dir)

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
    type: native
    display_name: Support Agent
    model: {default_model}
    output_type: SupportResult
    system_prompt:
      - You are a support agent
      - "Context: {{data}}"
  researcher:
    type: native
    display_name: Research Agent
    model: {default_model}
    output_type: ResearchResult
    system_prompt: You are a researcher
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
    agent_1 = NativeAgentConfig(name="agent1", model="test")
    agent_2 = NativeAgentConfig(name="agent2", model="test")
    return AgentsManifest(agents={"agent1": agent_1, "agent2": agent_2})


@pytest.fixture
async def pool(manifest):
    """Create test pool with agents."""
    async with AgentPool(manifest) as pool:
        yield pool
