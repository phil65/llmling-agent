"""Storage configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from platformdirs import user_data_dir
from pydantic import ConfigDict, Field
from pydantic_ai import Agent
from schemez import Schema


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


LogFormat = Literal["chronological", "conversations"]
FilterMode = Literal["and", "override"]
SupportedFormats = Literal["yaml", "toml", "json", "ini"]
FormatType = SupportedFormats | Literal["auto"]

APP_NAME: Final = "llmling-agent"
APP_AUTHOR: Final = "llmling"
DATA_DIR: Final = Path(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_DB_NAME: Final = "history.db"
DEFAULT_TITLE_PROMPT = """\
Create a short & consise title for this message. Only answer with that title.
"""


def get_database_path() -> str:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


class BaseStorageProviderConfig(Schema):
    """Base storage provider configuration."""

    type: str = Field(init=False)
    """Storage provider type."""

    log_messages: bool = Field(
        default=True,
        title="Log messages",
    )
    """Whether to log messages"""

    agents: set[str] | None = Field(
        default=None,
        title="Agent filter",
    )
    """Optional set of agent names to include. If None, logs all agents."""

    log_conversations: bool = Field(
        default=True,
        title="Log conversations",
    )
    """Whether to log conversations"""

    log_commands: bool = Field(
        default=True,
        title="Log commands",
    )
    """Whether to log command executions"""

    log_context: bool = Field(
        default=True,
        title="Log context",
    )
    """Whether to log context messages."""

    model_config = ConfigDict(frozen=True)


class SQLStorageConfig(BaseStorageProviderConfig):
    """SQL database storage configuration."""

    type: Literal["sql"] = Field("sql", init=False)
    """SQLModel storage configuration."""

    url: str = Field(
        default_factory=get_database_path,
        examples=["sqlite:///history.db", "postgresql://user:pass@localhost/db"],
        title="Database URL",
    )
    """Database URL (e.g. sqlite:///history.db)"""

    pool_size: int = Field(
        default=5,
        ge=1,
        examples=[5, 10, 20],
        title="Connection pool size",
    )
    """Connection pool size"""

    auto_migration: bool = Field(
        default=True,
        title="Auto migration",
    )
    """Whether to automatically add missing columns"""

    def get_engine(self) -> AsyncEngine:
        from sqlalchemy.ext.asyncio import create_async_engine

        # Convert URL to async format
        url_str = str(self.url)
        if url_str.startswith("sqlite://"):
            url_str = url_str.replace("sqlite://", "sqlite+aiosqlite://", 1)
        elif url_str.startswith("postgresql://"):
            url_str = url_str.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url_str.startswith("mysql://"):
            url_str = url_str.replace("mysql://", "mysql+aiomysql://", 1)

        # SQLite doesn't support pool_size parameter
        if url_str.startswith("sqlite+aiosqlite://"):
            return create_async_engine(url_str)
        return create_async_engine(url_str, pool_size=self.pool_size)


class TextLogConfig(BaseStorageProviderConfig):
    """Text log configuration."""

    type: Literal["text_file"] = Field("text_file", init=False)
    """Text log storage configuration."""

    path: str = Field(
        examples=["/var/log/agent.log", "~/logs/conversations.txt"],
        title="Log file path",
    )
    """Path to log file"""

    format: LogFormat = Field(
        default="chronological",
        examples=["chronological", "conversations"],
        title="Log format",
    )
    """Log format template to use"""

    template: Literal["chronological", "conversations"] | str | None = Field(  # noqa: PYI051
        default="chronological",
        examples=["chronological", "conversations", "/path/to/template.j2"],
        title="Template",
    )
    """Template to use: either predefined name or path to custom template"""

    encoding: str = Field(
        default="utf-8",
        examples=["utf-8", "ascii", "latin1"],
        title="File encoding",
    )
    """File encoding"""


# Config:
class FileStorageConfig(BaseStorageProviderConfig):
    """File storage configuration."""

    type: Literal["file"] = Field("file", init=False)
    """File storage configuration."""

    path: str = Field(
        examples=["/data/storage.json", "~/agent_data.yaml"],
        title="Storage file path",
    )
    """Path to storage file (extension determines format unless specified)"""

    format: FormatType = Field(
        default="auto",
        examples=["auto", "json", "yaml", "toml"],
        title="Storage format",
    )
    """Storage format (auto=detect from extension)"""

    encoding: str = Field(
        default="utf-8",
        examples=["utf-8", "ascii", "latin1"],
        title="File encoding",
    )
    """File encoding"""


class MemoryStorageConfig(BaseStorageProviderConfig):
    """In-memory storage configuration for testing."""

    type: Literal["memory"] = Field("memory", init=False)
    """In-memory storage configuration for testing."""


StorageProviderConfig = Annotated[
    SQLStorageConfig | FileStorageConfig | TextLogConfig | MemoryStorageConfig,
    Field(discriminator="type"),
]


class StorageConfig(Schema):
    """Global storage configuration."""

    providers: list[StorageProviderConfig] | None = Field(
        default=None,
        title="Storage providers",
    )
    """List of configured storage providers"""

    default_provider: str | None = Field(
        default=None,
        examples=["sql", "file", "memory"],
        title="Default provider",
    )
    """Name of default provider for history queries.
    If None, uses first configured provider."""

    agents: set[str] | None = Field(
        default=None,
        title="Global agent filter",
    )
    """Global agent filter. Can be overridden by provider-specific filters."""

    filter_mode: FilterMode = Field(
        default="and",
        examples=["and", "override"],
        title="Filter mode",
    )
    """How to combine global and provider agent filters:
    - "and": Both global and provider filters must allow the agent
    - "override": Provider filter overrides global filter if set
    """

    log_messages: bool = Field(
        default=True,
        title="Log messages",
    )
    """Whether to log messages."""

    log_conversations: bool = Field(
        default=True,
        title="Log conversations",
    )
    """Whether to log conversations."""

    log_commands: bool = Field(
        default=True,
        title="Log commands",
    )
    """Whether to log command executions."""

    log_context: bool = Field(
        default=True,
        title="Log context",
    )
    """Whether to log additions to the context."""

    title_generation_model: str = Field(
        default="openai:gpt-5-nano",
        examples=["openai:gpt-5-nano", "anthropic:claude-3-haiku"],
        title="Title generation model",
    )
    """Whether to log command executions."""

    title_generation_prompt: str = Field(
        default=DEFAULT_TITLE_PROMPT,
        examples=[DEFAULT_TITLE_PROMPT, "Summarize this conversation in 5 words"],
        title="Title generation prompt",
    )
    """Whether to log additions to the context."""

    model_config = ConfigDict(frozen=True)

    @property
    def effective_providers(self) -> list[StorageProviderConfig]:
        """Get effective list of providers.

        Returns:
            - Default SQLite provider if providers is None
            - Empty list if providers is empty list
            - Configured providers otherwise
        """
        if self.providers is None:
            if os.getenv("PYTEST_CURRENT_TEST"):
                return [MemoryStorageConfig()]
            return [SQLStorageConfig()]
        return self.providers

    def get_title_generation_agent(self) -> Agent:
        """Get title generation agent configuration."""
        return Agent(
            name="title_generation",
            model=self.title_generation_model,
            instructions=self.title_generation_prompt,
        )
