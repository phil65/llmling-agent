"""Function discovery and metadata handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.utils.inspection import get_fn_name


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.common_types import AnyCallable

logger = get_logger(__name__)


@dataclass
class NodeFunction:
    """Metadata for a function that uses nodes."""

    func: AnyCallable
    """The actual function to execute."""

    depends_on: list[str] = field(default_factory=list)
    """Names of functions this one depends on."""

    deps: Any = None
    """Node dependencies."""

    default_inputs: dict[str, Any] = field(default_factory=dict)
    """Default parameter values."""

    name: str = field(init=False)
    """Function name (from function.__name__)."""

    def __post_init__(self) -> None:
        """Set name and validate dependencies."""
        self.name = get_fn_name(self.func)
        # Extract default inputs from function signature

        sig = inspect.signature(self.func)
        self.default_inputs = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not param.empty
        }
        logger.debug("Registered node function", name=self.name, deps=self.depends_on)


def node_function(
    func: Callable[..., Any] | None = None,
    *,
    deps: Any | None = None,
    depends_on: str | Sequence[str | Callable[..., Any]] | Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Mark a function for automatic node execution.

    Can be used as simple decorator or with arguments:

    @node_function
    async def func(): ...

    @node_function(order=1, depends_on="other_func")
    async def func(): ...

    Args:
        func: Function to mark
        deps: Dependencies to inject into all Agent parameters
        depends_on: Names of functions this one depends on

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        match depends_on:
            case None:
                depends_on_ = []
            case str():
                depends_on_ = [depends_on]
            case Callable():
                depends_on_ = [get_fn_name(depends_on)]
            case [*items]:
                depends_on_ = [
                    i.__name__ if isinstance(i, Callable) else str(i)  # type: ignore[union-attr, arg-type]
                    for i in items
                ]
            case _:
                msg = f"Invalid depends_on: {depends_on}"
                raise ValueError(msg)
        # TODO: we still need to inject the deps in execution part.
        metadata = NodeFunction(func=func, depends_on=depends_on_ or [], deps=deps)
        func._node_function = metadata  # type: ignore
        return func

    return decorator(func) if func is not None else decorator
