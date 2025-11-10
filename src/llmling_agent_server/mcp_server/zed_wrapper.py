"""Function parameter encoding utils for Zed compatibility."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.prompts.prompts import DynamicPrompt
from llmling_agent.utils.importing import import_callable


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.prompts.prompts import BasePrompt


logger = get_logger(__name__)


def decode_zed_args(input_str: str) -> dict[str, Any]:
    """Decode Zed-style input string into kwargs dict."""
    if " :: " not in input_str:
        # Single argument case - use as first argument
        return {"main_arg": input_str}

    # Multiple arguments case
    parts = input_str.split(" :: ", 1)
    result: dict[str, Any] = {"main_arg": parts[0]}

    if len(parts) > 1:
        for pair in parts[1].split(" | "):
            if not pair:
                continue
            key, value = pair.split("=", 1)
            # Convert value types
            match value.lower():
                case "true":
                    result[key] = True
                case "false":
                    result[key] = False
                case "null":
                    result[key] = None
                case _:
                    try:
                        if "." in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except ValueError:
                        result[key] = value

    return result


def create_zed_wrapper(func: Any) -> Any:
    """Create a wrapper that accepts Zed-style input."""

    @wraps(func)
    def wrapper(input: str, **_kwargs: Any) -> Any:  # noqa: A002
        kwargs = decode_zed_args(input)
        return func(**kwargs)

    # Copy over metadata
    wrapper.__module__ = func.__module__
    wrapper.__qualname__ = f"zed_wrapped_{func.__qualname__}"
    return wrapper


def prepare_prompts_for_zed(prompts: Sequence[BasePrompt]) -> list[BasePrompt]:
    """Prepare prompts for Zed compatibility."""
    result: list[BasePrompt] = []

    for prompt in prompts:
        # Only process DynamicPrompts
        if not isinstance(prompt, DynamicPrompt):
            result.append(prompt)
            continue

        # Skip if only one or zero parameters
        # if len(prompt.arguments) <= 1:
        #     result.append(prompt)
        #     continue

        try:
            # Get and wrap the original function
            original_func = (
                import_callable(prompt.import_path)
                if isinstance(prompt.import_path, str)
                else prompt.import_path
            )
            wrapped_func = create_zed_wrapper(original_func)

            # Create new prompt with single input parameter
            new_prompt = DynamicPrompt(
                name=prompt.name,
                description=prompt.description,
                import_path=f"{wrapped_func.__module__}.{wrapped_func.__qualname__}",
                template=prompt.template,
                completions=prompt.completions,
            )
            result.append(new_prompt)
            logger.debug("Wrapped prompt for Zed mode", name=prompt.name)

        except Exception:
            logger.exception("Failed to wrap function for prompt", name=prompt.name)
            result.append(prompt)

    return result
