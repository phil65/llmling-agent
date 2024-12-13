from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from typer.testing import CliRunner

from llmling_agent.cli.agent import agent_cli


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def setup_environment():
    """Disable logging and mock config store for all tests."""
    # Disable logging
    logging.getLogger("llmling_agent").setLevel(logging.CRITICAL)
    logging.getLogger("yamling").setLevel(logging.CRITICAL)

    # Mock ConfigStore
    with mock.patch("llmling_agent.cli.agent.agent_store") as mock_store:
        # Setup basic mock behavior
        mock_store.get_active.return_value = None
        mock_store.get_config.side_effect = KeyError("Not found")
        mock_store.list_configs.return_value = []
        yield


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a minimal test config file."""
    config = {
        "responses": {
            "BasicResult": {
                "description": "Simple test result",
                "fields": {
                    "success": {"type": "bool"},
                    "message": {"type": "str"},
                },
            }
        },
        "agents": {
            "test_agent": {
                "name": "test_agent",
                "model": "openai:gpt-3.5-turbo",
                "result_type": "BasicResult",
                "system_prompts": ["You are a test agent."],
                "user_prompts": ["Hello!"],
            }
        },
    }
    config_path = tmp_path / "test_config.yml"
    config_path.write_text(json.dumps(config))
    return config_path


def test_list_agents_command(config_file: Path) -> None:
    """Test that list command runs and returns expected format."""
    result = runner.invoke(agent_cli, ["list", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "test_agent" in result.stdout


@mock.patch("llmling_agent.cli.agent.agent_store")
def test_add_agent_command(
    mock_store: mock.MagicMock, tmp_path: Path, config_file: Path
) -> None:
    """Test that add command runs successfully."""
    result = runner.invoke(agent_cli, ["add", "test", str(config_file)])
    assert result.exit_code == 0
    assert "Added agent configuration" in result.stdout
    mock_store.add_config.assert_called_once_with("test", str(config_file))


@mock.patch("llmling_agent.cli.agent.agent_store")
def test_set_agent_command(mock_store: mock.MagicMock, config_file: Path) -> None:
    """Test that set command runs."""
    # Configure mock to simulate existing config
    mock_store.get_config.return_value = str(config_file)

    result = runner.invoke(agent_cli, ["set", "test"])
    assert result.exit_code == 0
    assert "Set 'test' as active" in result.stdout
    mock_store.set_active.assert_called_once_with("test")


def test_help_commands() -> None:
    """Test that help commands work."""
    commands = ["list", "add", "set", "run"]
    for cmd in commands:
        result = runner.invoke(agent_cli, [cmd, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout


# Create runner once
runner = CliRunner()
if __name__ == "__main__":
    pytest.main([__file__])