"""Event handler configuration models for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.common_types import IndividualEventHandler


class BaseEventHandlerConfig(Schema):
    """Base configuration for event handlers."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str = Field(init=False)
    """Event handler type discriminator."""

    enabled: bool = Field(default=True)
    """Whether this handler is enabled."""

    def get_handler(self) -> IndividualEventHandler:
        """Create and return the configured event handler.

        Returns:
            Configured event handler callable.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class BuiltinEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for built-in event handlers (simple, detailed)."""

    type: Literal["builtin"] = Field("builtin", init=False)
    """Builtin event handler."""

    handler: Literal["simple", "detailed"] = Field(
        default="simple",
        examples=["simple", "detailed"],
    )
    """Which builtin handler to use.

    - simple: Basic text and tool notifications
    - detailed: Comprehensive execution visibility
    """

    def get_handler(self) -> IndividualEventHandler:
        """Get the builtin event handler."""
        from llmling_agent.agent.builtin_handlers import (
            detailed_print_handler,
            simple_print_handler,
        )

        handlers = {
            "simple": simple_print_handler,
            "detailed": detailed_print_handler,
        }
        return handlers[self.handler]


class CallbackEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for custom callback event handlers via import path."""

    type: Literal["callback"] = Field("callback", init=False)
    """Callback event handler."""

    import_path: str = Field(
        examples=[
            "mymodule:my_handler",
            "mypackage.handlers:custom_event_handler",
        ],
    )
    """Import path to the handler function (module:function format)."""

    def get_handler(self) -> IndividualEventHandler:
        """Import and return the callback handler."""
        from llmling_agent.utils.importing import import_callable

        return import_callable(self.import_path)


EventHandlerConfig = Annotated[
    BuiltinEventHandlerConfig | CallbackEventHandlerConfig,
    Field(discriminator="type"),
]


def resolve_handler_configs(
    configs: Sequence[EventHandlerConfig] | None,
) -> list[IndividualEventHandler]:
    """Resolve event handler configs to actual handler callables.

    Args:
        configs: List of event handler configurations.

    Returns:
        List of resolved event handler callables.
    """
    if not configs:
        return []
    return [cfg.get_handler() for cfg in configs]
