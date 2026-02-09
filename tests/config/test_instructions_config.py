"""Test ProviderInstructionConfig model."""

from __future__ import annotations

import pytest


class TestProviderInstructionConfig:
    """Test ProviderInstructionConfig model."""

    async def test_valid_config_with_ref(self):
        """Test valid config with ref field only."""
        from agentpool_config.instructions import ProviderInstructionConfig

        config = ProviderInstructionConfig(
            ref="my_provider",
        )

        assert config.type == "provider"
        assert config.ref == "my_provider"
        assert config.import_path is None
        assert config.kw_args == {}

    async def test_valid_config_with_import_path(self):
        """Test valid config with import_path field only."""
        from agentpool_config.instructions import ProviderInstructionConfig

        config = ProviderInstructionConfig(
            import_path="my.module.Provider",
        )

        assert config.type == "provider"
        assert config.import_path == "my.module.Provider"
        assert config.ref is None
        assert config.kw_args == {}

    async def test_valid_config_with_import_path_and_kwargs(self):
        """Test valid config with import_path and kwargs."""
        from agentpool_config.instructions import ProviderInstructionConfig

        config = ProviderInstructionConfig(
            import_path="my.module.Provider",
            kw_args={"key": "value"},
        )

        assert config.type == "provider"
        assert config.import_path == "my.module.Provider"
        assert config.ref is None
        assert config.kw_args == {"key": "value"}

    async def test_invalid_config_both_ref_and_import_path(self):
        """Test that config with both ref and import_path raises error."""
        from agentpool_config.instructions import ProviderInstructionConfig

        with pytest.raises(ValueError, match="Only one of 'ref' or 'import_path' can be provided"):
            ProviderInstructionConfig(
                ref="my_provider",
                import_path="my.module.Provider",
            )

    async def test_invalid_config_neither_ref_nor_import_path(self):
        """Test that config with neither ref nor import_path raises error."""
        from agentpool_config.instructions import ProviderInstructionConfig

        with pytest.raises(ValueError, match="Either 'ref' or 'import_path' must be provided"):
            ProviderInstructionConfig()
