"""Agent provider implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling import ToolError

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from pydantic import BaseModel


logger = get_logger(__name__)


async def get_structured_response(
    model_cls: type[BaseModel], use_promptantic: bool = True
) -> BaseModel:
    if use_promptantic:
        from promptantic import ModelGenerator, PromptanticError

        try:
            return await ModelGenerator().apopulate(model_cls)
        except PromptanticError as e:
            logger.exception("Failed to get structured input")
            error_msg = f"Invalid input: {e}"
            raise ToolError(error_msg) from e
        except KeyboardInterrupt:
            msg = "Input cancelled by user"
            raise ToolError(msg)  # noqa: B904
    else:
        # Regular text input
        print(f"(Please provide response as {model_cls.__name__})")
        response = input("> ")
        try:
            return model_cls.model_validate_json(response)
        except Exception as e:
            logger.exception("Failed to parse structured response")
            error_msg = f"Invalid response format: {e}"
            raise ToolError(error_msg) from e
