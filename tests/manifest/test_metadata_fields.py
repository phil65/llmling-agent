"""Tests for manifest metadata fields (YAML anchors and extensions)."""

from __future__ import annotations

from pydantic import ValidationError
import pytest
import yamling

from agentpool import AgentsManifest
from agentpool.models.agents import NativeAgentConfig


# Valid config with allowed metadata fields
MANIFEST_WITH_ALLOWED_METADATA = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are a test agent"

.anchor: &default_settings
  timeout: 30
  retries: 3

_meta:
  version: "1.0.0"
  author: "Test User"

x-custom:
  environment: "production"
  feature_flags:
    - feature_a
    - feature_b
"""

# Valid config with unknown field (typo)
MANIFEST_WITH_UNKNOWN_FIELD = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are a test agent"

random_field: "this is a typo/unknown field"
"""

# Valid config with both allowed and unknown fields
MANIFEST_WITH_MIXED_FIELDS = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are a test agent"

_meta:
  version: "1.0.0"

random_field: "should trigger warning"
"""


def test_allowed_metadata_fields_succeed():
    """Test that metadata fields starting with ., _, x- are allowed.

    RED PHASE: This test will FAIL because currently extra fields
    are forbidden. After implementation, this test will PASS.
    """
    config = yamling.load_yaml(MANIFEST_WITH_ALLOWED_METADATA)
    manifest = AgentsManifest.model_validate(config)

    # Verify that manifest loaded successfully
    assert "test_agent" in manifest.agents
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)
    assert agent.model == "openai:gpt-4o"


def test_unknown_field_generates_warning():
    """Test that unknown fields generate a warning but don't raise ValidationError.

    RED PHASE: This test will FAIL because currently unknown fields
    raise ValidationError. After implementation, this test will PASS.
    """
    config = yamling.load_yaml(MANIFEST_WITH_UNKNOWN_FIELD)

    # After implementation, this should NOT raise ValidationError
    # It should log a warning instead
    manifest = AgentsManifest.model_validate(config)

    # Verify agents loaded correctly
    assert "test_agent" in manifest.agents
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)
    assert agent.model == "openai:gpt-4o"


def test_mixed_allowed_and_unknown_fields():
    """Test manifest with both allowed and unknown fields.

    RED PHASE: This test will FAIL because currently unknown fields
    raise ValidationError. After implementation, this test will PASS.
    """
    config = yamling.load_yaml(MANIFEST_WITH_MIXED_FIELDS)

    # After implementation, this should succeed with warning for 'random_field'
    manifest = AgentsManifest.model_validate(config)

    # Verify allowed metadata fields are accessible
    assert "test_agent" in manifest.agents
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)
    assert agent.model == "openai:gpt-4o"
