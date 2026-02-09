"""Instruction provider wrapper for config-based dynamic instructions."""

from __future__ import annotations


__all__ = ["InstructionProvider"]

from typing import TYPE_CHECKING, Any, Literal

from agentpool.log import get_logger
from agentpool.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from agentpool.prompts.instructions import InstructionFunc
    from agentpool_config.instructions import ProviderInstructionConfig

logger = get_logger(__name__)


class InstructionProvider(ResourceProvider):
    """Provider wrapper for ProviderInstructionConfig.

    This provider resolves instruction functions from either:
    1. A reference to an existing provider (ref)
    2. An import path to instantiate a provider (import_path)

    When instructions are requested, it delegates to the referenced
    provider's get_instructions() method.
    """

    kind: Literal["custom"] = "custom"

    def __init__(
        self,
        config: ProviderInstructionConfig,
        toolsets: list[ResourceProvider] | None = None,
    ) -> None:
        """Initialize instruction provider.

        Args:
            config: The ProviderInstructionConfig to wrap
            toolsets: List of existing toolsets to search for ref resolution
        """
        super().__init__(name=f"instruction:{config.ref or config.import_path}")
        self.config = config
        self.toolsets = toolsets or []

    async def get_tools(self) -> list[Any]:
        """Return empty - this is instructions-only."""
        return []

    async def get_instructions(self) -> list[InstructionFunc]:
        """Resolve and return instruction functions.

        For ref: Find the referenced provider in toolsets and delegate.
        For import_path: Instantiate the provider and delegate.
        """
        from agentpool.utils.importing import import_callable

        if self.config.ref:
            # Find referenced provider in toolsets by name
            for provider in self.toolsets:
                if provider.name == self.config.ref and isinstance(provider, ResourceProvider):
                    logger.info(
                        "Delegating to referenced provider",
                        ref=self.config.ref,
                        provider=provider.__class__.__name__,
                    )
                    return await provider.get_instructions()
            logger.warning(
                "Referenced provider not found in toolsets",
                ref=self.config.ref,
                available_providers=[p.name for p in self.toolsets],
            )
            return []

        if self.config.import_path:
            # Instantiate provider from import path
            instructions: list[InstructionFunc] = []
            try:
                provider_cls = import_callable(self.config.import_path)
                provider_instance = provider_cls(**self.config.kw_args)
                if isinstance(provider_instance, ResourceProvider):
                    logger.info(
                        "Instantiating provider from import path",
                        import_path=self.config.import_path,
                        provider=provider_cls.__name__,
                    )
                    instructions = await provider_instance.get_instructions()
                else:
                    logger.warning(
                        "Instantiated provider does not implement get_instructions",
                        import_path=self.config.import_path,
                        provider=provider_cls.__name__,
                    )
            except (ImportError, TypeError, AttributeError):
                logger.exception(
                    "Failed to instantiate provider from import path",
                    import_path=self.config.import_path,
                )
            return instructions

        return []
