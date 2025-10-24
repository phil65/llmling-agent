"""Tests for environment configuration types."""

from __future__ import annotations

import pathlib
from typing import Any

from llmling import Config
from pydantic import ValidationError
import pytest
import yamling

from llmling_agent_config.environment import FileEnvironment, InlineEnvironment


@pytest.fixture
def sample_config() -> Config:
    """Create a sample Config for testing."""
    return Config()


def test_file_environment_basic():
    """Test basic file environment creation."""
    env = FileEnvironment(uri="config.yml")
    assert env.type == "file"
    assert env.uri == "config.yml"
    assert env.get_display_name() == "File: config.yml"


def test_file_environment_path_resolution(tmp_path: pathlib.Path):
    """Test path resolution relative to config file."""
    # Create a mock config structure
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    env_file = config_dir / "env.yml"

    # Create minimal valid YAML content
    env_file.write_text(yamling.dump_yaml({}))

    # Test relative path resolution
    path = str(config_dir / "agent.yml")
    env = FileEnvironment(uri="env.yml", config_file_path=path)
    resolved = env.get_file_path()
    assert pathlib.Path(resolved).is_absolute()
    assert pathlib.Path(resolved).parent == config_dir


def test_file_environment_validation():
    """Test validation rules for file environments."""
    # URI is required
    with pytest.raises(ValidationError):
        FileEnvironment()  # type: ignore

    # Empty URI
    with pytest.raises(ValidationError):
        FileEnvironment(uri="")


def test_inline_environment_basic(sample_config: Config):
    """Test basic inline environment creation."""
    env = InlineEnvironment.from_config(sample_config, uri="default-tools")
    assert env.type == "inline"
    assert env.uri == "default-tools"
    assert env.get_display_name() == "Inline: default-tools"

    # Test without URI
    env = InlineEnvironment.from_config(sample_config)
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
                "config": {},
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
):
    """Test environment type creation and path handling."""
    # Validate using the specific model
    env = expected_type.model_validate(env_data)
    assert isinstance(env, expected_type)
    assert env.get_file_path() == expected_path


def test_environment_display_names():
    """Test display name generation for different environment types."""
    file_env = FileEnvironment(uri="config.yml")
    assert file_env.get_display_name() == "File: config.yml"
    cfg = Config()
    inline_env = InlineEnvironment.from_config(cfg, uri="custom-env")
    assert inline_env.get_display_name() == "Inline: custom-env"

    # Without URI
    inline_env = InlineEnvironment()
    assert inline_env.get_display_name() == "Inline configuration"


def test_environment_serialization(sample_config: Config):
    """Test environment serialization."""
    # File environment
    file_env = FileEnvironment(uri="config.yml")
    data = file_env.model_dump()
    assert data["type"] == "file"
    assert data["uri"] == "config.yml"

    # Inline environment
    inline_env = InlineEnvironment.from_config(sample_config)
    data = inline_env.model_dump()
    assert data["type"] == "inline"
