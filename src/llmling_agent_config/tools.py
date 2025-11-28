"""Models for tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool


class BaseToolConfig(Schema):
    """Base configuration for agent tools."""

    type: str = Field(init=False)
    """Type discriminator for tool configs."""

    name: str | None = Field(
        default=None,
        examples=["search_web", "file_reader", "calculator"],
        title="Tool name override",
    )
    """Optional override for the tool name."""

    description: str | None = Field(
        default=None,
        examples=["Search the web for information", "Read file contents"],
        title="Tool description override",
    )
    """Optional override for the tool description."""

    enabled: bool = Field(default=True, title="Tool enabled")
    """Whether this tool is initially enabled."""

    requires_confirmation: bool = Field(default=False, title="Requires confirmation")
    """Whether tool execution needs confirmation."""

    metadata: dict[str, str] = Field(
        default_factory=dict,
        examples=[
            {"category": "web", "version": "1.0"},
            {"author": "system", "tags": "search,utility"},
            {"priority": "high", "environment": "production"},
        ],
        title="Tool metadata",
    )
    """Additional tool metadata."""

    hints: ToolHints | None = Field(
        default=None,
        title="Execution hints",
        examples=[
            {"read_only": True, "destructive": False, "open_world": True, "idempotent": False},
        ],
    )
    """Hints for tool execution."""

    model_config = ConfigDict(frozen=True)

    def get_tool(self) -> Tool:
        """Convert config to Tool instance."""
        raise NotImplementedError


class ToolHints(Schema):
    """Configuration for tool execution hints."""

    read_only: bool | None = Field(default=None, title="Read-only operation")
    """Hints that this tool only reads data without modifying anything"""

    destructive: bool | None = Field(default=None, title="Destructive operation")
    """Hints that this tool performs destructive operations that cannot be undone"""

    idempotent: bool | None = Field(default=None, title="Idempotent operation")
    """Hints that this tool has idempotent behaviour"""

    open_world: bool | None = Field(default=None, title="External resource access")
    """Hints that this tool can access / interact with external resources beyond the
    current system"""


class ImportToolConfig(BaseToolConfig):
    """Configuration for importing tools from Python modules."""

    type: Literal["import"] = Field("import", init=False)
    """Import path based tool."""

    import_path: ImportString[Callable[..., Any]] = Field(
        examples=["webbrowser:open", "builtins:print"],
        title="Import path",
    )
    """Import path to the tool function."""

    def get_tool(self) -> Tool:
        """Import and create tool from configuration."""
        from llmling_agent.tools.base import Tool

        return Tool.from_callable(
            self.import_path,
            name_override=self.name,
            description_override=self.description,
            enabled=self.enabled,
            requires_confirmation=self.requires_confirmation,
            metadata=self.metadata,
        )


class CrewAIToolConfig(BaseToolConfig):
    """Configuration for CrewAI-based tools."""

    type: Literal["crewai"] = Field("crewai", init=False)
    """CrewAI tool configuration."""

    import_path: ImportString[Any] = Field(
        examples=["crewai_tools.BrowserTool", "crewai_tools.SearchTool"],
        title="CrewAI tool import path",
    )
    """Import path to CrewAI tool class."""

    params: dict[str, Any] = Field(default_factory=dict, title="Tool parameters")
    """Tool-specific parameters."""

    def get_tool(self) -> Tool:
        """Import and create CrewAI tool."""
        from llmling_agent.tools.base import Tool

        try:
            return Tool.from_crewai_tool(
                self.import_path(**self.params),
                name_override=self.name,
                description_override=self.description,
                enabled=self.enabled,
                requires_confirmation=self.requires_confirmation,
                metadata={"type": "crewai", **self.metadata},
            )
        except ImportError as e:
            raise ImportError("CrewAI not installed.") from e


class LangChainToolConfig(BaseToolConfig):
    """Configuration for LangChain tools."""

    type: Literal["langchain"] = Field("langchain", init=False)
    """LangChain tool configuration."""

    tool_name: str = Field(
        examples=["wikipedia", "google-search", "python_repl"],
        title="LangChain tool name",
    )
    """Name of LangChain tool to use."""

    params: dict[str, Any] = Field(default_factory=dict, title="Tool parameters")
    """Tool-specific parameters."""

    def get_tool(self) -> Tool:
        """Import and create LangChain tool."""
        try:
            from langchain.tools import load_tool  # type: ignore[import-not-found]

            from llmling_agent.tools.base import Tool

            return Tool.from_langchain_tool(
                load_tool(self.tool_name, **self.params),
                name_override=self.name,
                description_override=self.description,
                enabled=self.enabled,
                requires_confirmation=self.requires_confirmation,
                metadata={"type": "langchain", **self.metadata},
            )
        except ImportError as e:
            raise ImportError("LangChain not installed.") from e


# Union type for tool configs
ToolConfig = Annotated[
    ImportToolConfig | CrewAIToolConfig | LangChainToolConfig,
    Field(discriminator="type"),
]
