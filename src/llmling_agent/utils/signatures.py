"""Signature utils."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable


logger = get_logger(__name__)


def create_modified_signature(
    fn_or_sig: Callable[..., Any] | inspect.Signature,
    *,
    remove: str | list[str] | None = None,
    inject: dict[str, type] | None = None,
) -> inspect.Signature:
    """Create a modified signature by removing specified parameters / injecting new ones.

    Args:
        fn_or_sig: The function or signature to modify.
        remove: The parameter(s) to remove.
        inject: The parameter(s) to inject.

    Returns:
        The modified signature.
    """
    sig = fn_or_sig if isinstance(fn_or_sig, inspect.Signature) else inspect.signature(fn_or_sig)
    rem_keys = [remove] if isinstance(remove, str) else remove or []
    new_params = [p for p in sig.parameters.values() if p.name not in rem_keys]
    if inject:
        injected_params = []
        for k, v in inject.items():
            injected_params.append(
                inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=v)
            )
        new_params = injected_params + new_params
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
