from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import agent, task, tool

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Iterator

    from llmling_agent.models.observability import TraceloopProviderConfig


P = ParamSpec("P")
R = TypeVar("R")


class TraceloopProvider(ObservabilityProvider):
    def __init__(self, config: TraceloopProviderConfig):
        self.config = config
        api_key = (
            config.api_key.get_secret_value()
            if config.api_key
            else os.getenv("TRACELOOP_API_KEY")
        )
        Traceloop.init(traceloop_sync_enabled=True, api_key=api_key)

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap a function with AgentOps tracking."""
        name = span_name or msg_template or func.__name__
        wrapped = task(name)(func)
        return cast(Callable[P, R], wrapped)

    def wrap_agent[T](self, kls: type[T], name: str) -> type[T]:
        """Wrap an agent class with AgentOps tracking."""
        if not isinstance(kls, type):
            msg = "AgentOps @track_agent can only be used with classes"
            raise TypeError(msg)
        # Only pass the name to AgentOps
        wrapped = agent(name)(kls)
        return cast(type[T], wrapped)

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with AgentOps tracking."""
        wrapped = tool(name)(func)
        return cast(Callable[..., T], wrapped)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create a Traceloop span for manual instrumentation."""
        yield
