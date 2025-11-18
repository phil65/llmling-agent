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


def create_modified_signature(
    fn_or_sig: Callable[..., Any] | inspect.Signature,
    *,
    remove: str | list[str] | None = None,
    inject: dict[str, type] | None = None,
) -> inspect.Signature:
    sig = (
        fn_or_sig
        if isinstance(fn_or_sig, inspect.Signature)
        else inspect.signature(fn_or_sig)
    )
    rem_keys = [remove] if isinstance(remove, str) else remove or []
    new_params = [p for p in sig.parameters.values() if p.name not in rem_keys]
    if inject:
        for k, v in inject.items():
            new_params.insert(
                0,
                inspect.Parameter(
                    k, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=v
                ),
            )
    return sig.replace(parameters=new_params)


def modify_signature(
    fn: Callable[..., Any],
    *,
    remove: str | list[str] | None = None,
    inject: dict[str, type] | None = None,
) -> None:
    new_sig = create_modified_signature(fn, remove=remove, inject=inject)
    update_signature(fn, new_sig)


def update_signature(fn: Callable[..., Any], signature: inspect.Signature) -> None:
    fn.__signature__ = signature  # type: ignore
    fn.__annotations__ = {
        name: param.annotation for name, param in signature.parameters.items()
    } | {"return": signature.return_annotation}
