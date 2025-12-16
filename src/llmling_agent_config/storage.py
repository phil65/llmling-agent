"""Storage configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from platformdirs import user_data_dir
from pydantic import ConfigDict, Field
from schemez import Schema
from tokonomics import ModelName
from yamling import FormatType


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from llmling_agent.sessions.store import SessionStore


LogFormat = Literal["chronological", "conversations"]
FilterMode = Literal["and", "override"]

APP_NAME: Final = "llmling-agent"
APP_AUTHOR: Final = "llmling"
DATA_DIR: Final = Path(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_DB_NAME: Final = "history.db"
DEFAULT_TITLE_PROMPT: Final = """\
Generate a short, descriptive title (3-7 words) for this request. \
Only respond with the title, nothing else."""


def get_database_path() -> str:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


class BaseStorageProviderConfig(Schema):
    """Base storage provider configuration."""

    type: str = Field(init=False)
    """Storage provider type."""

    log_messages: bool = Field(default=True, title="Log messages")
    """Whether to log messages"""

    agents: set[str] | None = Field(default=None, title="Agent filter")
    """Optional set of agent names to include. If None, logs all agents."""

    log_conversations: bool = Field(default=True, title="Log conversations")
    """Whether to log conversations"""

    log_commands: bool = Field(default=True, title="Log commands")
    """Whether to log command executions"""

    log_context: bool = Field(default=True, title="Log context")
    """Whether to log context messages."""


class SQLStorageConfig(BaseStorageProviderConfig):
    """SQL database storage configuration."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "SQL Storage"})

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

    auto_migration: bool = Field(default=True, title="Auto migration")
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

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Text Log"})

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

    encoding: str = Field(default="utf-8", examples=["utf-8", "ascii"], title="File encoding")
    """File encoding"""


class FileStorageConfig(BaseStorageProviderConfig):
    """File storage configuration."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "File Storage"})

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

    encoding: str = Field(default="utf-8", examples=["utf-8", "ascii"], title="File encoding")
    """File encoding of the storage file."""


class MemoryStorageConfig(BaseStorageProviderConfig):
    """In-memory storage configuration for testing."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Memory Storage"})

    type: Literal["memory"] = Field("memory", init=False)
    """In-memory storage configuration for testing."""


StorageProviderConfig = Annotated[
    SQLStorageConfig | FileStorageConfig | TextLogConfig | MemoryStorageConfig,
    Field(discriminator="type"),
]


SessionStoreType = Literal["sql", "memory"]


class StorageConfig(Schema):
    """Global storage configuration.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/storage_configuration/
    """

    providers: list[StorageProviderConfig] | None = Field(
        default=None,
        title="Storage providers",
        examples=[[{"type": "file", "path": "/data/storage.json"}]],
    )
    """List of configured storage providers"""

    session_store: SessionStoreType = Field(
        default="sql",
        title="Session store type",
        examples=["sql", "memory"],
    )
    """Type of session store to use for session persistence.

    - "sql": Persist sessions to SQL database (uses same URL as SQL provider)
    - "memory": In-memory storage (sessions lost on restart)
    """

    default_provider: str | None = Field(
        default=None,
        examples=["sql", "file", "memory"],
        title="Default provider",
    )
    """Name of default provider for history queries.
    If None, uses first configured provider."""

    agents: set[str] | None = Field(default=None, title="Global agent filter")
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

    log_messages: bool = Field(default=True, title="Log messages")
    """Whether to log messages."""

    log_conversations: bool = Field(default=True, title="Log conversations")
    """Whether to log conversations."""

    log_commands: bool = Field(default=True, title="Log commands")
    """Whether to log command executions."""

    log_context: bool = Field(default=True, title="Log context")
    """Whether to log additions to the context."""

    title_generation_model: ModelName | str | None = Field(
        default="google-gla:gemini-2.5-flash-lite",
        examples=["google-gla:gemini-2.5-flash-lite", None],
        title="Title generation model",
    )
    """Model to use for generating conversation titles.
    Set to None to disable automatic title generation."""

    title_generation_prompt: str = Field(
        default=DEFAULT_TITLE_PROMPT,
        examples=[DEFAULT_TITLE_PROMPT, "Summarize this given request in 5 words"],
        title="Title generation prompt",
    )
    """Prompt template for generating conversation titles."""

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

    def get_session_store(self) -> SessionStore:
        """Create session store based on configuration."""
        from llmling_agent.sessions.store import MemorySessionStore
        from llmling_agent_storage.session_store import SQLSessionStore

        match self.session_store:
            case "memory":
                return MemorySessionStore()
            case "sql":
                # Find SQL config or use default
                sql_config = None
                for provider in self.effective_providers:
                    if isinstance(provider, SQLStorageConfig):
                        sql_config = provider
                        break
                if sql_config is None:
                    sql_config = SQLStorageConfig()
                return SQLSessionStore(sql_config)
