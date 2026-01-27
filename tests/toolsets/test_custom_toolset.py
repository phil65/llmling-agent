"""Tests for CustomToolsetConfig parameter passing support."""

from __future__ import annotations

from agentpool.resource_providers import ResourceProvider
from agentpool_config.toolsets import CustomToolsetConfig


class MockProvider(ResourceProvider):
    """Mock provider that accepts arbitrary kwargs."""

    def __init__(self, name: str, owner: str | None = None, **kwargs) -> None:
        """Initialize mock provider with arbitrary parameters."""
        super().__init__(name=name, owner=owner)
        self.custom_params = kwargs


async def test_custom_toolset_parameters():
    """Test that CustomToolsetConfig passes parameters to provider constructor."""
    # This should fail until we implement the 'parameters' field
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.MockProvider",
        parameters={"key": "value", "another": 123},
    )
    provider = config.get_provider()

    # Provider should have received the custom parameters
    assert hasattr(provider, "custom_params")
    assert provider.custom_params == {"key": "value", "another": 123}


async def test_custom_toolset_name_collision():
    """Test that parameters can override the default name."""
    # This should fail until we implement the 'parameters' field
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.MockProvider",
        parameters={"name": "new_name"},
    )
    provider = config.get_provider()

    # Provider should use the overridden name, not the class name
    assert provider.name == "new_name"
