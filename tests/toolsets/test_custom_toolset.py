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


class StrictProvider(ResourceProvider):
    """Mock provider with strict parameter requirements."""

    def __init__(self, name: str, required_arg: int):
        """Initialize strict provider with required argument."""
        super().__init__(name=name)
        self.required_arg = required_arg


async def test_custom_toolset_parameters():
    """Test that CustomToolsetConfig passes parameters to provider constructor."""
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
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.MockProvider",
        parameters={"name": "new_name"},
    )
    provider = config.get_provider()

    # Provider should use the overridden name, not the class name
    assert provider.name == "new_name"


async def test_custom_toolset_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    import pytest

    # Missing required_arg should raise TypeError
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.StrictProvider",
        parameters={"unknown_param": "value"},  # Missing required_arg
    )
    with pytest.raises(TypeError):
        config.get_provider()
