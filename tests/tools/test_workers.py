from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import AgentPool, AgentsManifest


if TYPE_CHECKING:
    from pathlib import Path

    from llmling_agent import Agent


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


def write_config(content: str, path: Path) -> Path:
    """Write config content to a file."""
    config_file = path / "agents.yml"
    config_file.write_text(content)
    return config_file


async def test_basic_worker_setup(tmp_path: Path):
    """Test basic worker registration and usage."""
    config_path = write_config(BASIC_WORKERS, tmp_path)
    manifest = AgentsManifest.from_file(config_path)

    async with AgentPool[None](manifest) as pool:
        main_agent: Agent[None] = pool.get_agent("main")

        # Verify workers were registered as tools
        assert "ask_worker" in main_agent.tools
        assert "ask_specialist" in main_agent.tools

        worker_tool = main_agent.tools["ask_worker"]
        assert worker_tool.source == "agent"
        assert worker_tool.metadata["agent"] == "worker"


async def test_history_sharing(tmp_path: Path):
    """Test history sharing between agents."""
    config_path = write_config(WORKERS_WITH_SHARING, tmp_path)
    manifest = AgentsManifest.from_file(config_path)
    async with AgentPool[None](manifest) as pool:
        main_agent = pool.get_agent("main")
        worker = pool.get_agent("worker")

        # Configure test models
        main_model = TestModel(call_tools=["ask_worker"])  # Only call worker
        worker_model = TestModel(custom_output_text="The value is 42")  # string response

        # Override the models
        main_agent.set_model(main_model)
        worker.set_model(worker_model)

        # Create some conversation history
        await main_agent.run("Remember X equals 42")

        # Worker should have access to history
        result = await main_agent.run("Ask worker: What is X?")
        assert "42" in result.data


async def test_context_sharing(tmp_path: Path):
    """Test context sharing between agents."""
    config_path = write_config(WORKERS_WITH_SHARING, tmp_path)
    manifest = AgentsManifest.from_file(config_path)
    async with AgentPool[None](manifest) as pool:
        main_agent = pool.get_agent("main", deps={"important_value": 123})
        specialist = pool.get_agent("specialist")

        # Configure test models
        main_model = TestModel(call_tools=["ask_specialist"])
        specialist_model = TestModel(custom_output_text="I can see context value: 123")

        main_agent.set_model(main_model)
        specialist.set_model(specialist_model)
        prompt = "Ask specialist: What's in the context?"
        result = await main_agent.run(prompt)
        assert "123" in result.data


async def test_invalid_worker(tmp_path: Path):
    """Test error when using non-existent worker."""
    config_path = write_config(INVALID_WORKERS, tmp_path)
    manifest = AgentsManifest.from_file(config_path)

    with pytest.raises(ValueError, match=r"Worker agent.*not found"):
        async with AgentPool[None](manifest):
            pass


async def test_worker_independence(tmp_path: Path):
    """Test that workers maintain independent state when not sharing."""
    config_path = write_config(BASIC_WORKERS, tmp_path)
    manifest = AgentsManifest.from_file(config_path)
    async with AgentPool[None](manifest) as pool:
        main_agent: Agent[None] = pool.get_agent("main")

        # Create history in main agent
        await main_agent.run("Remember X equals 42")

        # Worker should not see this history
        result = await main_agent.run("Ask worker: What is X?")
        assert "42" not in result.data


async def test_multiple_workers_same_prompt(tmp_path: Path):
    """Test using multiple workers with the same prompt."""
    config_path = write_config(BASIC_WORKERS, tmp_path)
    manifest = AgentsManifest.from_file(config_path)
    async with AgentPool[None](manifest) as pool:
        main_agent: Agent[None] = pool.get_agent("main")
        worker: Agent[None] = pool.get_agent("worker")
        specialist: Agent[None] = pool.get_agent("specialist")

        # Configure test models
        main_model = TestModel(call_tools=["ask_worker", "ask_specialist"])
        worker_model = TestModel(custom_output_text="I am a helpful worker assistant")
        specialist_model = TestModel(custom_output_text="I am a domain specialist")

        main_agent.set_model(main_model)
        worker.set_model(worker_model)
        specialist.set_model(specialist_model)

        responses = []
        main_agent.message_sent.connect(lambda msg: responses.append(msg.content))

        await main_agent.run("Ask both workers: introduce yourselves")

        assert len(responses) > 0
        assert any("helpful worker" in r.lower() for r in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
