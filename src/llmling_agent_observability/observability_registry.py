from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.models.observability import (
    AgentOpsProviderConfig,
    ArizePhoenixProviderConfig,
    BaseObservabilityProviderConfig,
    LangsmithProviderConfig,
    LogfireProviderConfig,
    ObservabilityConfig,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent_observability.base_provider import ObservabilityProvider

logger = get_logger(__name__)


@dataclass
class TrackedDecoration:
    """Registration of an item (callable / class) that needs observability tracking.

    These registrations are permanent and collected at import time through decorators.
    Each new provider will use these registrations to apply its tracking.

    The system is a bit more "sophisticated" because everything is working lazily in
    order to not having to import the observability platforms (and thus, also the AI
    libraries) on library loading.
    """

    target: Any  # class or function to be decorated
    name: str
    kwargs: dict[str, Any]


class ObservabilityRegistry:
    """Registry for pending decorations and provider configuration."""

    def __init__(self) -> None:
        self._registered_agents: dict[str, TrackedDecoration] = {}
        self._registered_tools: dict[str, TrackedDecoration] = {}
        self._registered_actions: dict[str, TrackedDecoration] = {}
        self.providers: list[ObservabilityProvider] = []
        # to prevent double registration
        self._registered_provider_classes: set[type[ObservabilityProvider]] = set()

    def register_agent(
        self,
        name: str,
        target: type[Any],
        **kwargs: Any,
    ) -> None:
        """Register a class for agent tracking."""
        self._registered_agents[name] = TrackedDecoration(
            target=target,
            name=name,
            kwargs=kwargs,
        )
        logger.debug("Registered agent %r for observability tracking", name)

    def register_tool(
        self,
        name: str,
        target: Callable,
        **kwargs: Any,
    ) -> None:
        """Register a function for tool tracking."""
        self._registered_tools[name] = TrackedDecoration(
            target=target,
            name=name,
            kwargs=kwargs,
        )
        logger.debug("Registered tool %r for observability tracking", name)

    def register_action(
        self,
        name: str,
        target: Callable,
        **kwargs: Any,
    ) -> None:
        """Register a function for action tracking."""
        self._registered_actions[name] = TrackedDecoration(
            target=target,
            name=name,
            kwargs=kwargs,
        )
        msg = "Registered action %r for observability tracking with args %r"
        logger.debug(msg, name, kwargs)

    def configure_provider(self, provider: ObservabilityProvider) -> None:
        """Configure a new provider and apply tracking to all registered items.

        When a new provider is configured, it will:
        1. Get added to the list of active providers
        2. Apply its tracking to all previously registered functions/tools/agents
        3. Be available for immediate tracking of new registrations

        The registry maintains a permanent list of what needs tracking,
        collected through decorators at import time. Each provider uses
        these registrations to know what to track.
        """
        logger.info(
            "Configuring provider: %s, Current pending actions: %s",
            provider.__class__.__name__,
            list(self._registered_actions.keys()),
        )
        self.providers.append(provider)

        # Apply decorations for each type
        for pending in self._registered_agents.values():
            try:
                pending.target = provider.wrap_agent(
                    pending.target,
                    pending.name,
                    **pending.kwargs,
                )
                logger.debug("Applied agent tracking to %r", pending.name)
            except Exception:
                msg = "Failed to apply agent tracking to %r"
                logger.exception(msg, pending.name)

        for pending in self._registered_tools.values():
            try:
                pending.target = provider.wrap_tool(
                    pending.target,
                    pending.name,
                    **pending.kwargs,
                )
                logger.debug("Applied tool tracking to %r", pending.name)
            except Exception:
                msg = "Failed to apply tool tracking to %r"
                logger.exception(msg, pending.name)

        for pending in self._registered_actions.values():
            try:
                pending.target = provider.wrap_action(
                    pending.target,
                    msg_template=pending.name,
                    **pending.kwargs,
                )
                msg = "Applied action tracking to %r with args %r"
                logger.debug(msg, pending.name, pending.kwargs)
            except Exception:
                msg = "Failed to apply action tracking to %r"
                logger.exception(msg, pending.name)

    def register_providers(self, observability_config: ObservabilityConfig) -> None:
        """Register and configure all observability providers.

        Args:
            observability_config: Configuration for observability providers
        """
        if not observability_config.enabled:
            return
        for library in observability_config.instrument_libraries or []:
            match library:
                case "pydantic_ai":
                    import pydantic_ai  # noqa
                case "litellm":
                    import litellm  # noqa

        # Configure each provider
        for provider_config in observability_config.providers:
            provider_cls = get_provider_cls(provider_config)
            if provider_cls not in self._registered_provider_classes:
                provider = provider_cls(provider_config)  # type: ignore
                logger.debug("Registering %s", provider_cls.__name__)
                self._registered_provider_classes.add(provider_cls)
                self.configure_provider(provider)


def get_provider_cls(
    provider_config: BaseObservabilityProviderConfig,
) -> type[ObservabilityProvider]:
    match provider_config:
        case LogfireProviderConfig():
            from llmling_agent_observability.logfire_provider import LogfireProvider

            return LogfireProvider

        case AgentOpsProviderConfig():
            from llmling_agent_observability.agentops_provider import AgentOpsProvider

            return AgentOpsProvider

        case LangsmithProviderConfig():
            from llmling_agent_observability.langsmith_provider import LangsmithProvider

            return LangsmithProvider

        case ArizePhoenixProviderConfig():
            from llmling_agent_observability.arize_provider import ArizePhoenixProvider

            return ArizePhoenixProvider
        case _:
            msg = f"Unknown provider config: {provider_config}"
            raise ValueError(msg)


registry = ObservabilityRegistry()
