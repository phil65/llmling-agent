from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TypeVar

from llmling_agent_observability.base_provider import ObservabilityProvider


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class MockCall:
    """Record of a provider call."""

    call_type: Literal["span", "wrap_agent", "wrap_tool", "wrap_action"]
    name: str
    kwargs: dict[str, Any]


@dataclass
class MockProvider(ObservabilityProvider):
    """Mock provider that collects calls for testing."""

    calls: list[MockCall] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
        """Record span creation."""
        self.calls.append(
            MockCall(
                call_type="span",
                name=name,
                kwargs={"attributes": attributes},
            )
        )
        yield

    def wrap_agent(self, func: Callable, name: str, **kwargs: Any) -> Callable:
        """Record agent wrapper creation."""
        self.calls.append(
            MockCall(
                call_type="wrap_agent",
                name=name,
                kwargs=kwargs,
            )
        )
        return func

    def wrap_tool(self, func: Callable, name: str, **kwargs: Any) -> Callable:
        """Record tool wrapper creation."""
        self.calls.append(
            MockCall(
                call_type="wrap_tool",
                name=name,
                kwargs=kwargs,
            )
        )
        return func

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
        **kwargs: Any,
    ) -> Callable[P, R]:
        """Record action wrapper creation."""
        self.calls.append(
            MockCall(
                call_type="wrap_action",
                name=span_name or msg_template or func.__name__,
                kwargs={"msg_template": msg_template, **kwargs},
            )
        )
        return func
