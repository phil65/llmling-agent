"""Embedding model configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, SecretStr
from schemez import Schema


class BaseEmbeddingConfig(Schema):
    """Base configuration for embedding models."""

    type: str = Field(init=False, title="Embedding model type")
    """Type identifier for the embedding model."""

    model_config = ConfigDict(frozen=True)


class SentenceTransformersConfig(BaseEmbeddingConfig):
    """Configuration for sentence-transformers models."""

    type: Literal["sentence-transformers"] = Field(default="sentence-transformers", init=False)

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        examples=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
        title="Model name",
    )
    """Name of the model to use."""

    use_gpu: bool = Field(default=False, title="Use GPU")
    """Whether to use GPU for inference."""

    batch_size: int = Field(default=32, ge=1, examples=[16, 32, 64], title="Batch size")
    """Batch size for inference."""


class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OpenAI's embedding API."""

    type: Literal["openai"] = Field(default="openai", init=False)

    model: str = Field(
        default="text-embedding-ada-002",
        examples=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        title="OpenAI model",
    )
    """Model to use."""

    api_key: SecretStr | None = Field(default=None, title="OpenAI API key")
    """OpenAI API key."""


class BGEConfig(BaseEmbeddingConfig):
    """Configuration for BGE embedding models."""

    type: Literal["bge"] = Field(default="bge", init=False)

    model_name: str = Field(
        default="BAAI/bge-small-en",
        examples=["BAAI/bge-small-en", "BAAI/bge-base-en", "BAAI/bge-large-en"],
        title="BGE model name",
    )
    """Name/size of BGE model to use."""

    use_gpu: bool = Field(default=False, title="Use GPU")
    """Whether to use GPU for inference."""

    batch_size: int = Field(default=32, ge=1, examples=[16, 32, 64], title="Batch size")
    """Batch size for inference."""


class LiteLLMEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for LiteLLM embeddings."""

    type: Literal["litellm"] = Field(default="litellm", init=False)

    model: str = Field(
        examples=["text-embedding-ada-002", "mistral/mistral-embed", "gemini/text-embedding-004"],
        title="LiteLLM model identifier",
    )
    """Model identifier (e.g., 'text-embedding-ada-002',
    'mistral/mistral-embed', 'gemini/text-embedding-004')."""

    api_key: SecretStr | None = Field(default=None, title="Provider API key")
    """API key for the provider."""

    dimensions: int | None = Field(
        default=None,
        examples=[512, 768, 1536],
        title="Embedding dimensions",
    )
    """Optional number of dimensions for the embeddings."""

    batch_size: int = Field(default=32, ge=1, examples=[16, 32, 64], title="Batch size")
    """Batch size for inference."""

    additional_params: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        title="Additional parameters",
    )
    """Additional parameters to pass to litellm.embedding()."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


# Union type for embedding configs
EmbeddingConfig = Annotated[
    SentenceTransformersConfig | OpenAIEmbeddingConfig | BGEConfig | LiteLLMEmbeddingConfig,
    Field(discriminator="type"),
]
