"""Token counting utilities with fallback strategies."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_TOKEN_MODEL = "gpt-3.5-turbo"


@lru_cache
def has_tiktoken() -> bool:
    """Check if tiktoken is available."""
    return bool(find_spec("tiktoken"))


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in text with fallback strategy.

    Args:
        text: Text to count tokens for
        model: Optional model name for tiktoken (ignored in fallback)

    Returns:
        Estimated token count
    """
    if has_tiktoken():
        import tiktoken

        encoding = tiktoken.encoding_for_model(model or DEFAULT_TOKEN_MODEL)
        return len(encoding.encode(text))

    # Fallback: very rough approximation
    # Strategies could be:
    # 1. ~4 chars per token (quick but rough)
    # 2. Word count * 1.3 (better for English)
    # 3. Split on common token boundaries
    return len(text.split()) + len(text) // 4


def batch_count_tokens(texts: Sequence[str], model: str | None = None) -> list[int]:
    """Count tokens for multiple texts.

    Args:
        texts: Sequence of texts to count
        model: Optional model name for tiktoken

    Returns:
        List of token counts
    """
    if has_tiktoken():
        import tiktoken

        encoding = tiktoken.encoding_for_model(model or DEFAULT_TOKEN_MODEL)
        return [len(encoding.encode(text)) for text in texts]

    return [count_tokens(text) for text in texts]
