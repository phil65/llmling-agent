"""Agent injection utilities."""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent import AgentPool, MessageNode


logger = get_logger(__name__)


def is_node_type(typ: Any) -> bool:
    """Check if a type is or inherits from MessageNode."""
    from llmling_agent import MessageNode

    # Direct type check
    if typ is MessageNode:
        return True

    # For actual types (not GenericAlias etc)
    if isinstance(typ, type):
        return issubclass(typ, MessageNode)

    # For generic types (Agent[T], StructuredAgent[T], etc)
    origin = getattr(typ, "__origin__", None)
    if origin is not None and isinstance(origin, type):
        return issubclass(origin, MessageNode)

    return False


class NodeInjectionError(Exception):
    """Raised when agent injection fails."""


def inject_nodes[T, **P](
    func: Callable[P, T],
    pool: AgentPool,
    provided_kwargs: dict[str, Any],
) -> dict[str, MessageNode[Any, Any]]:
    """Get nodes to inject based on function signature."""
    hints = typing.get_type_hints(func)
    params = inspect.signature(func).parameters
    msg = "Injecting nodes for %s.%s"
    logger.debug(msg, func.__module__, func.__qualname__)
    logger.debug("Type hints: %s", hints)
    logger.debug("Available nodes in pool: %s", sorted(pool.nodes))

    nodes: dict[str, MessageNode[Any, Any]] = {}
    for name, param in params.items():
        if param.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            logger.debug("Skipping %s: wrong parameter kind %s", name, param.kind)
            continue

        hint = hints.get(name)
        if hint is None:
            logger.debug("Skipping %s: no type hint", name)
            continue

        # Handle Optional/Union types
        origin = getattr(hint, "__origin__", None)
        args = getattr(hint, "__args__", ())

        # Check for MessageNode or any of its subclasses
        is_node = (
            is_node_type(hint)  # Direct node type
            or (  # Optional[Node[T]] or Union containing Node
                origin is not None and any(is_node_type(arg) for arg in args)
            )
        )

        if not is_node:
            msg = "Skipping %s: not a node type (hint=%s, origin=%s, args=%s)"
            logger.debug(msg, name, hint, origin, args)
            continue

        logger.debug("Found node parameter: %s", name)

        # Check for duplicate parameters
        if name in provided_kwargs and provided_kwargs[name] is not None:
            msg = (
                f"Cannot inject node {name!r}: Parameter already provided.\n"
                f"Remove the explicit argument or rename the parameter."
            )
            logger.error(msg)
            raise NodeInjectionError(msg)

        # Get node from pool
        if name not in pool.nodes:
            available = ", ".join(sorted(pool.nodes))
            msg = (
                f"No node named {name!r} found in pool.\n"
                f"Available nodes: {available}\n"
                f"Check your YAML configuration or node name."
            )
            logger.error(msg)
            raise NodeInjectionError(msg)

        nodes[name] = pool.nodes[name]
        logger.debug("Injecting node %s for parameter %s", nodes[name], name)

    logger.debug("Injection complete. Injected nodes: %s", sorted(nodes))
    return nodes
