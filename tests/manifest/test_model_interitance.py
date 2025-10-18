from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling_agent import AgentsManifest


if TYPE_CHECKING:
    from pathlib import Path


BASIC_INHERITANCE = """\
agents:
    base:
        model: {default_model}
        name: Base Agent
        system_prompts:
            - "Base prompt"

    child:
        inherits: base
        name: Child Agent
        system_prompts:
            - "Child prompt"
"""


MULTI_LEVEL = """\
agents:
    base:
        model: {default_model}
        name: Base

    middle:
        inherits: base
        system_prompts:
            - "Middle prompt"

    child:
        inherits: middle
        name: Child
        model: {default_model}
"""


CIRCULAR = """\
agents:
    agent1:
        inherits: agent2
        model: {default_model}

    agent2:
        inherits: agent1
        model: {default_model}
"""


INVALID_PARENT = """\
agents:
    child:
        inherits: nonexistent
        model: {default_model}
"""


@pytest.fixture
def basic_config(tmp_path: Path, default_model: str) -> Path:
    """Create config with basic inheritance."""
    config_file = tmp_path / "agents.yml"
    config_file.write_text(BASIC_INHERITANCE.format(default_model=default_model))
    return config_file


@pytest.fixture
def multi_level_config(tmp_path: Path, default_model: str) -> Path:
    """Create config with multi-level inheritance."""
    config_file = tmp_path / "multi.yml"
    config_file.write_text(MULTI_LEVEL.format(default_model=default_model))
    return config_file


@pytest.fixture
def circular_config(tmp_path: Path, default_model: str) -> Path:
    """Create config with circular inheritance."""
    config_file = tmp_path / "circular.yml"
    config_file.write_text(CIRCULAR.format(default_model=default_model))
    return config_file


@pytest.fixture
def invalid_parent_config(tmp_path: Path, default_model: str) -> Path:
    """Create config with invalid parent reference."""
    config_file = tmp_path / "invalid.yml"
    config_file.write_text(INVALID_PARENT.format(default_model=default_model))
    return config_file


def test_basic_inheritance(basic_config: Path, default_model: str):
    """Test basic parent-child inheritance."""
    manifest = AgentsManifest.from_file(basic_config)

    child = manifest.agents["child"]
    assert child.name == "Child Agent"
    assert child.model.identifier == default_model  # type: ignore
    assert child.system_prompts == ["Child prompt"]


def test_multi_level_inheritance(multi_level_config: Path, default_model: str):
    """Test inheritance through multiple levels."""
    manifest = AgentsManifest.from_file(multi_level_config)

    child = manifest.agents["child"]
    assert child.name == "Child"
    assert child.model.identifier == default_model  # type: ignore
    assert child.system_prompts == ["Middle prompt"]  # Inherited from middle


def test_circular_inheritance(circular_config: Path, default_model: str):
    """Test that circular inheritance is detected."""
    with pytest.raises(ValueError, match="Circular inheritance"):
        AgentsManifest.from_yaml(CIRCULAR.format(default_model=default_model))


def test_invalid_parent(invalid_parent_config: Path, default_model: str):
    """Test error on invalid parent reference."""
    with pytest.raises(ValueError, match=r"Parent agent.*not found"):
        AgentsManifest.from_yaml(INVALID_PARENT.format(default_model=default_model))


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--log-level", "debug"])
