"""Model capability utilities."""

from __future__ import annotations


async def supports_vision(model_name: str | None) -> bool:
    """Check if a model supports vision capabilities.

    Args:
        model_name: Name of the model to check

    Returns:
        True if the model supports vision, False otherwise
    """
    if not model_name:
        return False

    try:
        import tokonomics

        caps = await tokonomics.get_model_capabilities(model_name)
        return bool(caps and caps.supports_vision)
    except ImportError:
        # If tokonomics is not available, return False
        return False
