"""Vector store configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic.functional_validators import model_validator


class BaseVectorStoreConfig(BaseModel):
    """Base configuration for vector stores."""

    type: str = Field(init=False)
    """Type identifier for the vector store."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class ChromaConfig(BaseVectorStoreConfig):
    """Configuration for ChromaDB vector store."""

    type: Literal["chroma"] = Field(default="chroma", init=False)

    persist_directory: str | None = None
    """Where to persist the database."""

    collection_name: str = "default"
    """Name of the collection to use."""


class QdrantConfig(BaseVectorStoreConfig):
    """Configuration for Qdrant vector store."""

    type: Literal["qdrant"] = Field(default="qdrant", init=False)

    location: str | None = None
    """Path to local Qdrant storage. If None, uses memory."""

    url: str | None = None
    """URL for Qdrant server. If set, location is ignored."""

    api_key: SecretStr | None = None
    """API key for Qdrant cloud."""

    collection_name: str = "default"
    """Name of the collection to use."""

    prefer_grpc: bool = True
    """Whether to prefer gRPC over HTTP."""

    @model_validator(mode="after")
    def validate_connection(self) -> QdrantConfig:
        """Ensure either location or url is set, but not both."""
        if self.location and self.url:
            msg = "Cannot specify both location and url"
            raise ValueError(msg)
        if self.api_key and not self.url:
            msg = "API key only valid with url"
            raise ValueError(msg)
        return self


VectorStoreConfig = Annotated[
    ChromaConfig | QdrantConfig,  # etc
    Field(discriminator="type"),
]
