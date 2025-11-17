"""Base classes for providers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Self

from pydantic import BaseModel


class BaseProvider[ConfigT: BaseModel]:
    """Base class for all providers."""

    def __init__(self, config: ConfigT) -> None:
        """Initialize provider with configuration."""
        self.config = config

    @classmethod
    @abstractmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Alternative constructor with explicit parameters."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return string representation of provider with non-default config values."""
        non_defaults = self.config.model_dump(exclude_defaults=True)
        fields_str = ", ".join(f"{k}={v!r}" for k, v in non_defaults.items())
        return f"{self.__class__.__name__}({fields_str})"
