"""Tests for agent configuration models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
import pytest
from schemez import InlineSchemaDef
import yamling

from llmling_agent import AgentsManifest


if TYPE_CHECKING:
    from pathlib import Path


VALID_AGENT_CONFIG = """\
responses:
  TestResponse:
    response_schema:
        description: Test response
        type: inline
        fields:
            message:
                type: str
                description: A message
            score:
                type: int
                ge: 0
                le: 100

agents:
  test_agent:  # Key is the agent ID
    name: Test Agent
    description: A test agent
    model: test
    result_type: TestResponse
    system_prompts:
      - You are a test agent
"""

INVALID_RESPONSE_CONFIG = """\
responses: {}
agent:
  name: Test Agent
  model: test
  result_type: NonExistentResponse
  system_prompts: []
"""


ENV_CONFIG = """\
{}
"""

ENV_AGENT = """\
responses:
    BasicResult:
       response_schema:
            description: Test result
            type: inline
            fields:
                message:
                    type: str
                    description: Test message

agents:
    test_agent:
        name: test
        model: test
        result_type: BasicResult
        environment: env.yml  # Relative path
"""


def test_valid_agent_definition():
    """Test valid complete agent configuration."""
    agent_def = AgentsManifest.model_validate(yamling.load_yaml(VALID_AGENT_CONFIG))
    schema = agent_def.responses["TestResponse"].response_schema
    assert isinstance(schema, InlineSchemaDef)
    score = schema.fields["score"]  # pyright: ignore
    assert score.ge == 0
    assert score.le == 100  # noqa: PLR2004


def test_missing_referenced_response():
    """Test referencing non-existent response model."""
    config = yamling.load_yaml(INVALID_RESPONSE_CONFIG)
    with pytest.raises(ValidationError):
        AgentsManifest.model_validate(config)


def test_environment_path_resolution(tmp_path: Path):
    """Test that environment paths are resolved relative to config file."""
    # Create a mock environment config with valid structure
    env_file = tmp_path / "env.yml"
    env_file.write_text(ENV_CONFIG)
    # Create agent config referencing the environment
    config_file = tmp_path / "agents.yml"
    config_file.write_text(ENV_AGENT)

    # Load the config and verify path resolution
    agent_def = AgentsManifest.from_file(config_file)
    test_agent = agent_def.agents["test_agent"]

    # The environment path should now be resolved
    config = test_agent.get_config()
    assert config is not None  # Just verify it loads without error
