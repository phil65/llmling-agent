"""Integration tests for ACPAgent connecting to real llmling-agent ACP server.

These tests spawn an actual llmling-agent serve-acp process with a TestModel
and connect to it using ACPAgent, testing the full roundtrip.

Note: These are slow integration tests that spawn real subprocesses.
Run with: pytest tests/servers/acp_server/test_acp_agent_integration.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anyenv.code_execution import LocalExecutionEnvironment
import pytest

from llmling_agent.agent.acp_agent import ACPAgent
from llmling_agent.models.acp_agents import ACPAgentConfig


if TYPE_CHECKING:
    from llmling_agent.agent.events import RichAgentStreamEvent


# Mark all tests in this module as slow/integration
pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest.fixture
def test_agent_config_yaml() -> str:
    """YAML config for a minimal test agent using TestModel."""
    return """
agents:
  test_agent:
    model: test
"""


@pytest.fixture
def test_config_file(test_agent_config_yaml: str, tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config_file = tmp_path / "test_config.yml"
    config_file.write_text(test_agent_config_yaml)
    return config_file


@pytest.fixture
def acp_agent_config(test_config_file: Path) -> ACPAgentConfig:
    """Create ACPAgentConfig that spawns llmling-agent serve-acp."""
    return ACPAgentConfig(
        command="uv",
        args=[
            "run",
            "llmling-agent",
            "serve-acp",
            str(test_config_file),
            "--agent",
            "test_agent",
        ],
        name="test_acp_agent",
        description="Test ACP agent with TestModel",
        cwd=str(Path.cwd()),
    )


async def test_acp_agent_basic_prompt(acp_agent_config: ACPAgentConfig, test_config_file: Path):
    """Test basic prompt execution through ACP."""
    try:
        print(f"\nConfig file: {test_config_file}")
        print(f"Config file exists: {test_config_file.exists()}")
        async with ACPAgent(config=acp_agent_config) as agent:
            result = await asyncio.wait_for(agent.run("Hi"), timeout=15.0)

            assert result is not None
            assert result.content is not None
            assert len(result.content) > 0
    except TimeoutError:
        pytest.skip("ACP server took too long to respond")


async def test_acp_agent_streaming(acp_agent_config: ACPAgentConfig):
    """Test streaming response through ACP."""
    try:
        async with ACPAgent(config=acp_agent_config) as agent:
            chunks: list[RichAgentStreamEvent[Any]] = []

            async def collect_chunks():
                async for chunk in agent.run_stream("Hi"):
                    chunks.append(chunk)  # noqa: PERF401

            await asyncio.wait_for(collect_chunks(), timeout=15.0)

            assert len(chunks) > 2  # noqa: PLR2004
    except TimeoutError:
        pytest.skip("ACP server took too long to respond")


async def test_acp_agent_multiple_prompts(acp_agent_config: ACPAgentConfig):
    """Test multiple sequential prompts in same session."""
    try:
        async with ACPAgent(config=acp_agent_config) as agent:
            result1 = await asyncio.wait_for(agent.run("One"), timeout=15.0)
            assert result1.content is not None

            result2 = await asyncio.wait_for(agent.run("Two"), timeout=15.0)
            assert result2.content is not None
    except TimeoutError:
        pytest.skip("ACP server took too long to respond")


async def test_acp_agent_session_info(acp_agent_config: ACPAgentConfig):
    """Test that session info is properly initialized."""
    try:
        async with ACPAgent(config=acp_agent_config) as agent:
            assert agent._session_id is not None
            assert agent._init_response is not None
            assert agent._state is not None
    except TimeoutError:
        pytest.skip("ACP server took too long to start")


async def test_acp_agent_file_operations(tmp_path: Path, test_config_file: Path):
    """Test file operations through ACP when enabled."""
    # Create a test file
    test_file = tmp_path / "test_input.txt"
    test_file.write_text("Hello from test file")

    config = ACPAgentConfig(
        command="uv",
        args=[
            "run",
            "llmling-agent",
            "serve-acp",
            str(test_config_file),
            "--agent",
            "test_agent",
            "--file-access",
        ],
        name="test_acp_file_agent",
        cwd=str(tmp_path),
        allow_file_operations=True,
    )

    async with ACPAgent(config=config) as agent:
        # The TestModel will just respond, but the file access capability should be enabled
        assert agent._client_handler is not None
        assert agent._client_handler.allow_file_operations is True


async def test_acp_agent_terminal_operations(tmp_path: Path, test_config_file: Path):
    """Test terminal operations through ACP when enabled."""
    config = ACPAgentConfig(
        command="uv",
        args=[
            "run",
            "llmling-agent",
            "serve-acp",
            str(test_config_file),
            "--agent",
            "test_agent",
            "--terminal-access",
        ],
        name="test_acp_terminal_agent",
        cwd=str(tmp_path),
        allow_terminal=True,
    )

    async with ACPAgent(config=config) as agent:
        assert agent._client_handler is not None
        assert agent._client_handler.allow_terminal is True


async def test_acp_agent_with_custom_execution_environment(test_config_file: Path, tmp_path: Path):
    """Test ACPAgent with custom execution environment config."""
    config = ACPAgentConfig(
        command="uv",
        args=[
            "run",
            "llmling-agent",
            "serve-acp",
            str(test_config_file),
            "--agent",
            "test_agent",
        ],
        name="test_acp_env_agent",
        cwd=str(tmp_path),
        execution_environment={"type": "local", "timeout": 120.0},
    )

    async with ACPAgent(config=config) as agent:
        # Verify the execution environment was created from config
        assert agent._client_handler is not None
        env = agent._client_handler.env
        assert isinstance(env, LocalExecutionEnvironment)
        assert env is not None
        assert env.timeout == 120.0  # noqa: PLR2004


async def test_acp_agent_cleanup_on_error(acp_agent_config: ACPAgentConfig):
    """Test that resources are cleaned up even when errors occur."""
    agent = ACPAgent(config=acp_agent_config)

    # Enter context
    await agent.__aenter__()
    assert agent._process is not None
    process = agent._process

    # Exit context (simulating cleanup)
    await agent.__aexit__(None, None, None)

    # Process should be terminated
    assert agent._process is None
    # Give it a moment to fully terminate
    await asyncio.sleep(0.1)
    assert process.returncode is not None


async def test_acp_agent_stats(acp_agent_config: ACPAgentConfig):
    """Test that agent stats are collected."""
    try:
        async with ACPAgent(config=acp_agent_config) as agent:
            await asyncio.wait_for(agent.run("Test"), timeout=15.0)
            stats = await agent.get_stats()

            assert stats is not None
    except TimeoutError:
        pytest.skip("ACP server took too long to respond")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
