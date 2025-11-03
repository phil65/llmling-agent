"""Environment configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from llmling import Config
from pydantic import ConfigDict, Field
from schemez import Schema


class FileEnvironment(Schema):
    """File-based environment configuration.

    Loads environment settings from external YAML files, supporting:
    - Reusable environment configurations
    - Separation of concerns
    - Environment sharing between agents
    - Version control of environment settings
    """

    type: Literal["file"] = Field("file", init=False)
    """File-based runtime config."""

    uri: str = Field(min_length=1)
    """"Path to environment file."""

    config_file_path: str | None = None
    """Path to agent config file for resolving relative paths"""

    model_config = ConfigDict(frozen=True)

    def get_file_path(self) -> str:
        """Get resolved file path."""
        from upath import UPath

        if self.config_file_path:
            base_dir = UPath(self.config_file_path).parent
            return str(base_dir / self.uri)
        return self.uri

    def get_config(self) -> Config:
        """Get runtime configuration."""
        return Config.from_file(self.get_file_path())


# We directly inherit from Config in order to save a level of indentation in YAML
class InlineEnvironment(Config):
    """Direct environment configuration without external files.

    Allows embedding complete environment settings directly in the agent
    configuration instead of referencing external files. Useful for:
    - Self-contained configurations
    - Testing and development
    - Simple agent setups
    """

    type: Literal["inline"] = Field("inline", init=False)
    """Inline-defined runtime config."""

    uri: str | None = None
    """Optional identifier for this configuration"""

    config_file_path: str | None = None
    """Path to agent config file for resolving relative paths"""

    model_config = ConfigDict(frozen=True)

    def get_file_path(self) -> str | None:
        """No file path for inline environments."""
        return None

    def get_config(self) -> Config:
        """Get runtime configuration."""
        return self

    @classmethod
    def from_config(
        cls,
        config: Config,
        uri: str | None = None,
        config_file_path: str | None = None,
    ) -> InlineEnvironment:
        """Create inline environment from config."""
        return cls(**config.model_dump(), uri=uri)


AgentEnvironment = Annotated[
    FileEnvironment | InlineEnvironment, Field(discriminator="type")
]
