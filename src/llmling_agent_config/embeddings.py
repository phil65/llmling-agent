"""Embedding model configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, SecretStr
from schemez import Schema


class BaseEmbeddingConfig(Schema):
    """Base configuration for embedding models."""

    type: str = Field(init=False)
    """Type identifier for the embedding model."""

    model_config = ConfigDict(frozen=True)


class SentenceTransformersConfig(BaseEmbeddingConfig):
    """Configuration for sentence-transformers models."""

    type: Literal["sentence-transformers"] = Field(
        default="sentence-transformers",
        init=False,
    )

    model_name: str = "all-MiniLM-L6-v2"
    """Name of the model to use."""

    use_gpu: bool = False
    """Whether to use GPU for inference."""

    batch_size: int = 32
    """Batch size for inference."""


class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OpenAI's embedding API."""

    type: Literal["openai"] = Field(default="openai", init=False)

    model: str = "text-embedding-ada-002"
    """Model to use."""

    api_key: SecretStr | None = None
    """OpenAI API key."""


class BGEConfig(BaseEmbeddingConfig):
    """Configuration for BGE embedding models."""

    type: Literal["bge"] = Field(default="bge", init=False)

    model_name: str = "BAAI/bge-small-en"
    """Name/size of BGE model to use."""

    use_gpu: bool = False
    """Whether to use GPU for inference."""

    batch_size: int = 32
    """Batch size for inference."""


class LiteLLMEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for LiteLLM embeddings."""

    type: Literal["litellm"] = Field(default="litellm", init=False)

    model: str
    """Model identifier (e.g., 'text-embedding-ada-002',
    'mistral/mistral-embed', 'gemini/text-embedding-004')."""

    api_key: SecretStr | None = None
    """API key for the provider."""

    dimensions: int | None = None
    """Optional number of dimensions for the embeddings."""

    batch_size: int = 32
    """Batch size for inference."""

    additional_params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    """Additional parameters to pass to litellm.embedding()."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


# Union type for embedding configs
EmbeddingConfig = Annotated[
    SentenceTransformersConfig | OpenAIEmbeddingConfig | BGEConfig | LiteLLMEmbeddingConfig,
    Field(discriminator="type"),
]
