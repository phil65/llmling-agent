"""Tests for environment configuration types."""

from __future__ import annotations

import pathlib
from typing import Any

from llmling import Config
from llmling.config.models import GlobalSettings
from pydantic import ValidationError
import pytest
import yaml

from llmling_agent.environment import FileEnvironment, InlineEnvironment


@pytest.fixture
def sample_config() -> Config:
    """Create a sample Config for testing."""
    return Config()


def test_file_environment_basic() -> None:
    """Test basic file environment creation."""
    env = FileEnvironment(uri="config.yml")
    assert env.type == "file"
    assert env.uri == "config.yml"
    assert env.get_display_name() == "File: config.yml"


def test_file_environment_path_resolution(tmp_path: pathlib.Path) -> None:
    """Test path resolution relative to config file."""
    # Create a mock config structure
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    env_file = config_dir / "env.yml"

    # Create valid YAML content
    env_content = {
        "global_settings": {
            "llm_capabilities": {
                "load_resource": False,
                "get_resources": False,
            }
        }
    }
    env_file.write_text(yaml.dump(env_content))

    # Test relative path resolution
    env = FileEnvironment(
        uri="env.yml",
        config_file_path=str(config_dir / "agent.yml"),
    )
    resolved = env.get_file_path()
    assert pathlib.Path(resolved).is_absolute()
    assert pathlib.Path(resolved).parent == config_dir


def test_file_environment_validation() -> None:
    """Test validation rules for file environments."""
    # URI is required
    with pytest.raises(ValidationError):
        FileEnvironment(type="file")  # type: ignore

    # Empty URI
    with pytest.raises(ValidationError):
        FileEnvironment(type="file", uri="")


def test_inline_environment_basic(sample_config: Config) -> None:
    """Test basic inline environment creation."""
    env = InlineEnvironment(
        uri="default-tools",
        config=sample_config,
    )
    assert env.type == "inline"
    assert env.uri == "default-tools"
    assert env.get_display_name() == "Inline: default-tools"

    # Test without URI
    env = InlineEnvironment(config=sample_config)
    assert env.get_display_name() == "Inline configuration"


@pytest.mark.parametrize(
    ("env_data", "expected_type", "expected_path"),
    [
        (
            {"type": "file", "uri": "config.yml"},
            FileEnvironment,
            "config.yml",
        ),
        (
            {
                "type": "inline",
                "config": {"global_settings": {"llm_capabilities": {}}},
            },
            InlineEnvironment,
            None,
        ),
    ],
)
def test_environment_types(
    env_data: dict[str, Any],
    expected_type: type[FileEnvironment | InlineEnvironment],
    expected_path: str | None,
) -> None:
    """Test environment type creation and path handling."""
    # Validate using the specific model
    env = expected_type.model_validate(env_data)
    assert isinstance(env, expected_type)
    assert env.get_file_path() == expected_path


def test_environment_display_names() -> None:
    """Test display name generation for different environment types."""
    file_env = FileEnvironment(uri="config.yml")
    assert file_env.get_display_name() == "File: config.yml"

    inline_env = InlineEnvironment(
        config=Config(global_settings=GlobalSettings()),
        uri="custom-env",
    )
    assert inline_env.get_display_name() == "Inline: custom-env"

    # Without URI
    inline_env = InlineEnvironment(config=Config(global_settings=GlobalSettings()))
    assert inline_env.get_display_name() == "Inline configuration"


def test_environment_serialization(sample_config: Config) -> None:
    """Test environment serialization."""
    # File environment
    file_env = FileEnvironment(uri="config.yml")
    data = file_env.model_dump()
    assert data["type"] == "file"
    assert data["uri"] == "config.yml"

    # Inline environment
    inline_env = InlineEnvironment(config=sample_config)
    data = inline_env.model_dump()
    assert data["type"] == "inline"
    assert "config" in data