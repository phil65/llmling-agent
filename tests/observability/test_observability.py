"""Tests for the simplified observability system using Logfire."""

import os
from unittest.mock import patch

from pydantic import SecretStr
import pytest

from llmling_agent.observability.observability_registry import ObservabilityRegistry
from llmling_agent_config.observability import (
    CustomObservabilityConfig,
    LangsmithObservabilityConfig,
    LogfireObservabilityConfig,
    ObservabilityConfig,
)


@pytest.fixture
def clean_env():
    """Clean OTEL environment variables before and after tests."""
    env_vars = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_RESOURCE_ATTRIBUTES",
    ]

    original_values = {}  # Store original values
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    for var, value in original_values.items():  # Restore original values
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@patch("logfire.configure")
def test_registry_configure_logfire(mock_configure, clean_env):
    """Test registry configuration with Logfire backend."""
    registry = ObservabilityRegistry()
    provider = LogfireObservabilityConfig(
        token=SecretStr("test_token"),
        service_name="test-service",
        environment="test",
    )
    config = ObservabilityConfig(enabled=True, provider=provider)
    registry.configure_observability(config)
    # Check that OTEL environment variables were set
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://logfire-us.pydantic.dev"
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert "Authorization=Bearer test_token" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "service.name=test-service" in os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    assert "deployment.environment.name=test" in os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    mock_configure.assert_called_once()


@patch("logfire.configure")
def test_registry_configure_langsmith(mock_configure, clean_env):
    """Test registry configuration with Langsmith backend."""
    registry = ObservabilityRegistry()
    provider = LangsmithObservabilityConfig(
        api_key=SecretStr("ls_key"),
        project_name="test-project",
        service_name="my-service",
    )
    config = ObservabilityConfig(enabled=True, provider=provider)
    registry.configure_observability(config)
    # Check environment variables
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://api.smith.langchain.com"
    assert "x-api-key=ls_key" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    # Check that Logfire was configured to NOT send to logfire
    mock_configure.assert_called_once()


@patch("logfire.configure")
def test_registry_configure_custom(mock_configure, clean_env):
    """Test registry configuration with custom backend."""
    registry = ObservabilityRegistry()
    endpoint = "https://my-otel-collector.com:4318"
    provider = CustomObservabilityConfig(
        endpoint=endpoint,
        headers={"X-API-Key": "custom_key"},
        service_name="custom-service",
        environment="prod",
    )
    config = ObservabilityConfig(enabled=True, provider=provider)
    registry.configure_observability(config)
    # Check environment variables
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == endpoint
    assert "X-API-Key=custom_key" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    mock_configure.assert_called_once()


@patch("logfire.configure")
def test_registry_disabled(mock_configure, clean_env):
    """Test that disabled observability doesn't configure anything."""
    registry = ObservabilityRegistry()
    config = ObservabilityConfig(enabled=False)
    registry.configure_observability(config)
    mock_configure.assert_not_called()
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ


@patch("logfire.configure")
def test_registry_no_provider(mock_configure, clean_env):
    """Test that no provider doesn't configure anything."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(enabled=True, provider=None)
    registry.configure_observability(config)

    mock_configure.assert_not_called()


@patch("logfire.configure")
def test_registry_double_configuration(mock_configure, clean_env):
    """Test that registry prevents double configuration."""
    registry = ObservabilityRegistry()
    provider = LogfireObservabilityConfig(token=SecretStr("test"))
    config = ObservabilityConfig(enabled=True, provider=provider)
    registry.configure_observability(config)  # Configure twice
    registry.configure_observability(config)
    assert mock_configure.call_count == 1  # Should only be called once


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
