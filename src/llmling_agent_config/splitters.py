"""Configuration models for text chunking."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator
from schemez import Schema


class BaseChunkerConfig(Schema):
    """Base configuration for text chunkers."""

    type: str = Field(init=False, title="Chunker type")
    """Type identifier for the chunker."""

    chunk_overlap: int = Field(default=200, ge=0, examples=[100, 200, 500], title="Chunk overlap")
    """Number of characters to overlap between chunks."""

    model_config = ConfigDict(frozen=True)


class LangChainChunkerConfig(BaseChunkerConfig):
    """Configuration for LangChain chunkers."""

    type: Literal["langchain"] = Field(default="langchain", init=False)
    """Langchain chunker configuration."""

    chunker_type: Literal["recursive", "markdown", "character"] = Field(
        default="recursive",
        examples=["recursive", "markdown", "character"],
        title="LangChain chunker type",
    )
    """Which LangChain chunker to use."""

    chunk_size: int = Field(default=1000, ge=1, examples=[500, 1000, 2000], title="Chunk size")
    """Target size of chunks."""


class MarkoChunkerConfig(BaseChunkerConfig):
    """Configuration for marko-based markdown chunker."""

    type: Literal["marko"] = Field(default="marko", init=False)
    """Marko chunker configuration."""

    split_on: Literal["headers", "paragraphs", "blocks"] = Field(
        default="headers",
        examples=["headers", "paragraphs", "blocks"],
        title="Split strategy",
    )
    """How to split the markdown."""

    min_header_level: int = Field(
        default=2,
        ge=1,
        le=6,
        examples=[1, 2, 3],
        title="Minimum header level",
    )
    """Minimum header level to split on (if splitting on headers)."""

    combine_small_sections: bool = Field(default=True, title="Combine small sections")
    """Whether to combine small sections with neighbors."""

    min_section_length: int = Field(
        default=100,
        ge=0,
        examples=[50, 100, 200],
        title="Minimum section length",
    )
    """Minimum length for a section before combining."""

    @model_validator(mode="after")
    def validate_section_length(self) -> MarkoChunkerConfig:
        """Ensure min_section_length is only used with combine_small_sections."""
        if not self.combine_small_sections and self.min_section_length > 0:
            raise ValueError("min_section_length only valid when combine_small_sections=True")
        return self


class LlamaIndexChunkerConfig(BaseChunkerConfig):
    """Configuration for LlamaIndex chunkers."""

    type: Literal["llamaindex"] = Field(default="llamaindex", init=False)
    """LlamaIndex chunker configuration."""

    chunker_type: Literal["sentence", "token", "fixed", "markdown"] = Field(
        default="markdown",
        examples=["sentence", "token", "fixed", "markdown"],
        title="LlamaIndex chunker type",
    )
    """Which LlamaIndex chunker to use."""

    chunk_size: int = Field(default=1000, ge=1, examples=[500, 1000, 2000], title="Chunk size")
    """Target size of chunks."""

    include_metadata: bool = Field(default=True, title="Include metadata")
    """Whether to include document metadata in chunks."""

    include_prev_next_rel: bool = Field(default=False, title="Include relationships")
    """Whether to track relationships between chunks."""


# Union type for all chunker configs
ChunkerConfig = Annotated[
    LangChainChunkerConfig | LlamaIndexChunkerConfig | MarkoChunkerConfig,
    Field(discriminator="type"),
]
