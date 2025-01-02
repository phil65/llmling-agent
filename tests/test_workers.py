from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import AgentPool, AgentsManifest


if TYPE_CHECKING:
    from pathlib import Path

    from llmling_agent.agent.agent import Agent


BASIC_WORKERS = """\
agents:
  main:
    model: test
    name: Main Agent
    workers:
      - worker
      - specialist
    system_prompts:
      - "You are the main agent. Use your workers to help with tasks."

  worker:
    model: test
    name: Basic Worker
    system_prompts:
      - "You are a helpful worker agent."

  specialist:
    model: test
    name: Domain Specialist
    system_prompts:
      - "You are a specialist with deep domain knowledge."
"""

WORKERS_WITH_SHARING = """\
agents:
  main:
    model: test
    name: Main Agent
    workers:
      - name: worker
        pass_message_history: true
      - name: specialist
        share_context: true

  worker:
    model: test
    name: History Worker
    system_prompts:
      - "You are a worker with conversation history."

  specialist:
    model: test
    name: Context Worker
    system_prompts:
      - "You are a worker with context access."
"""

INVALID_WORKERS = """\
agents:
  main:
    model: test
    name: Main Agent
    workers:
      - nonexistent
"""


@pytest.fixture
def basic_config(tmp_path: Path) -> Path:
    """Create a temporary config file with basic worker setup."""
    config_file = tmp_path / "agents.yml"
    config_file.write_text(BASIC_WORKERS)
    return config_file


@pytest.fixture
def sharing_config(tmp_path: Path) -> Path:
    """Create a temporary config file with history/context sharing."""
    config_file = tmp_path / "sharing.yml"
    config_file.write_text(WORKERS_WITH_SHARING)
    return config_file


@pytest.fixture
def invalid_config(tmp_path: Path) -> Path:
    """Create a temporary config file with invalid worker setup."""
    config_file = tmp_path / "invalid.yml"
    config_file.write_text(INVALID_WORKERS)
    return config_file


async def test_basic_worker_setup(basic_config: Path):
    """Test basic worker registration and usage."""
    manifest = AgentsManifest[Any, Any].from_file(basic_config)

    async with AgentPool.open(manifest) as pool:
        main_agent: Agent[Any, str] = pool.get_agent("main")

        # Verify workers were registered as tools
        assert "ask_worker" in main_agent.tools
        assert "ask_specialist" in main_agent.tools

        worker_tool = main_agent.tools["ask_worker"]
        assert worker_tool.source == "agent"
        assert worker_tool.metadata["agent"] == "worker"


async def test_history_sharing(sharing_config: Path):
    """Test history sharing between agents."""
    manifest = AgentsManifest[Any, Any].from_file(sharing_config)

    async with AgentPool.open(manifest) as pool:
        main_agent = pool.get_agent("main")
        worker = pool.get_agent("worker")

        # Configure test models
        main_model = TestModel(
            call_tools=["ask_worker"],  # Only call worker
        )
        worker_model = TestModel(
            custom_result_text="The value is 42"  # Simple string response
        )

        # Override the models
        main_agent.set_model(main_model)
        worker.set_model(worker_model)

        # Create some conversation history
        await main_agent.run("Remember X equals 42")

        # Worker should have access to history
        result = await main_agent.run("Ask worker: What is X?")
        assert "42" in result.data


async def test_context_sharing(sharing_config: Path):
    """Test context sharing between agents."""
    manifest = AgentsManifest[Any, Any].from_file(sharing_config)

    async with AgentPool.open(manifest) as pool:
        main_agent = pool.get_agent("main")
        specialist = pool.get_agent("specialist")

        # Configure test models
        main_model = TestModel(
            call_tools=["ask_specialist"],  # Only call specialist
        )
        specialist_model = TestModel(
            custom_result_text="I can see context value: 123"  # Indicate context access
        )

        main_agent.set_model(main_model)
        specialist.set_model(specialist_model)

        context_data = {"important_value": 123}
        result = await main_agent.run(
            "Ask specialist: What's in the context?", deps=context_data
        )
        assert "123" in result.data


async def test_invalid_worker(invalid_config: Path):
    """Test error when using non-existent worker."""
    manifest = AgentsManifest[Any, Any].from_file(invalid_config)

    with pytest.raises(ValueError, match="Worker agent.*not found"):
        async with AgentPool.open(manifest):
            pass


async def test_worker_independence(basic_config: Path):
    """Test that workers maintain independent state when not sharing."""
    manifest = AgentsManifest[Any, Any].from_file(basic_config)

    async with AgentPool.open(manifest) as pool:
        main_agent: Agent[Any, str] = pool.get_agent("main")

        # Create history in main agent
        await main_agent.run("Remember X equals 42")

        # Worker should not see this history
        result = await main_agent.run("Ask worker: What is X?")
        assert "42" not in result.data


async def test_multiple_workers_same_prompt(basic_config: Path):
    """Test using multiple workers with the same prompt."""
    manifest = AgentsManifest[Any, Any].from_file(basic_config)

    async with AgentPool.open(manifest) as pool:
        main_agent: Agent[Any, str] = pool.get_agent("main")
        worker: Agent[Any, str] = pool.get_agent("worker")
        specialist: Agent[Any, str] = pool.get_agent("specialist")

        # Configure test models
        main_model = TestModel(
            call_tools=["ask_worker", "ask_specialist"],  # Call both workers
        )
        worker_model = TestModel(custom_result_text="I am a helpful worker assistant")
        specialist_model = TestModel(custom_result_text="I am a domain specialist")

        main_agent.set_model(main_model)
        worker.set_model(worker_model)
        specialist.set_model(specialist_model)

        responses = []
        main_agent.message_sent.connect(lambda msg: responses.append(msg.content))

        await main_agent.run("Ask both workers: introduce yourselves")

        assert len(responses) > 0
        assert any("helpful worker" in r.lower() for r in responses)
