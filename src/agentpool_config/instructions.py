"""Configuration models for instruction providers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ProviderInstructionConfig(BaseModel):
    """Configuration for provider-based dynamic instructions."""

    type: Literal["provider"] = Field("provider", init=False)

    ref: str | None = Field(
        default=None,
        description="Name of existing toolset provider to reference.",
    )

    import_path: str | None = Field(
        default=None,
        description="Python import path to ResourceProvider class.",
    )

    kw_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments to pass to the provider constructor.",
    )

    @model_validator(mode="after")
    def validate_ref_or_import_path(self) -> ProviderInstructionConfig:
        """Validate that exactly one of ref or import_path is provided."""
        if self.ref is None and self.import_path is None:
            raise ValueError("Either 'ref' or 'import_path' must be provided")
        if self.ref is not None and self.import_path is not None:
            raise ValueError("Only one of 'ref' or 'import_path' can be provided")
        return self
