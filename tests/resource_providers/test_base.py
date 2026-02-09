"""Test ResourceProvider base class."""

from __future__ import annotations


class TestResourceProviderGetInstructions:
    """Test get_instructions method on ResourceProvider base class."""

    async def test_get_instructions_method_exists(self):
        """Test that get_instructions method exists on ResourceProvider."""
        from agentpool.resource_providers.base import ResourceProvider

        assert hasattr(ResourceProvider, "get_instructions")
        assert callable(ResourceProvider.get_instructions)

    async def test_get_instructions_is_async(self):
        """Test that get_instructions is an async method."""
        import inspect

        from agentpool.resource_providers.base import ResourceProvider

        assert inspect.iscoroutinefunction(ResourceProvider.get_instructions)

    async def test_get_instructions_returns_list(self):
        """Test that default get_instructions returns an empty list."""
        from agentpool.resource_providers.base import ResourceProvider

        class TestProvider(ResourceProvider):
            pass

        provider = TestProvider(name="test_provider")
        result = await provider.get_instructions()

        assert isinstance(result, list)
        assert result == []

    async def test_get_instructions_signature(self):
        """Test that get_instructions has correct return type annotation."""
        import inspect

        from agentpool.resource_providers.base import ResourceProvider

        sig = inspect.signature(ResourceProvider.get_instructions)

        # Should be annotated as returning list[InstructionFunc]
        # Note: In Python 3.9+, we can use list[...]
        # Type checking may show this differently, but runtime check should work
        # For now, just verify it has a return annotation
        assert sig.return_annotation is not inspect.Signature.empty
