"""Signature utils."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable


logger = get_logger(__name__)


def _create_tool_signature_with_context(
    base_signature: inspect.Signature, key: str = "ctx"
) -> inspect.Signature:
    """Create a function signature that includes RunContext as first parameter.

    This is crucial for PydanticAI integration - it expects tools that need RunContext
    to have it as the first parameter with proper annotation. Without this, PydanticAI
    won't pass the RunContext and we lose access to tool_call_id and other context.

    Args:
        base_signature: Original signature from MCP tool schema (tool parameters only)
        key: Name of the parameter to add RunContext to

    Returns:
        New signature: (ctx: RunContext, ...original_params) -> ReturnType

    Example:
        Original: (message: str) -> str
        Result:   (ctx: RunContext, message: str) -> str
    """
    # Create RunContext parameter
    ctx_param = inspect.Parameter(
        key, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext
    )
    # Combine with tool parameters
    tool_params = list(base_signature.parameters.values())
    new_params = [ctx_param, *tool_params]

    return base_signature.replace(parameters=new_params)


def _create_tool_annotations_with_context(
    base_annotations: dict[str, Any],
    key: str = "ctx",
) -> dict[str, Any]:
    """Create function annotations that include RunContext for first parameter.

    Args:
        base_annotations: Original annotations from MCP tool schema
        key: Name of the parameter to add RunContext to
    Returns:
        New annotations dict with 'ctx': RunContext added to base annotations

    Example:
        Original: {'message': str, 'return': str}
        Result:   {'ctx': RunContext, 'message': str, 'return': str}
    """
    new_annotations = base_annotations.copy()
    new_annotations[key] = RunContext
    return new_annotations


def insert_arg_into_signature(fn: Callable, key: str, type_: type) -> Callable:
    sig = inspect.signature(fn)
    params = [
        inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=type_),
        *sig.parameters.values(),
    ]
    new_sig = inspect.Signature(params)
    fn.__signature__ = new_sig  # type: ignore
    fn.__annotations__ = dict(new_sig.parameters) | {"return": new_sig.return_annotation}
    return fn


def get_signature_without_argument(
    fn: Callable, key: str | list[str]
) -> inspect.Signature:
    # Hide AgentContext parameter from pydantic-ai's signature analysis
    sig = inspect.signature(fn)
    keys = [key] if isinstance(key, str) else key
    new_params = [p for p in sig.parameters.values() if p.name not in keys]
    return sig.replace(parameters=new_params)  # type: ignore
