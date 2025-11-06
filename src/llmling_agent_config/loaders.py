"""Models for resource information."""

from __future__ import annotations

from collections.abc import Sequence as TypingSequence
import inspect
from typing import Annotated, Literal
import warnings

from pydantic import ConfigDict, Field, SecretStr, model_validator
from schemez import Schema

from llmling_agent.common_types import JsonObject
from llmling_agent.utils.importing import import_callable


class BaseResourceLoaderConfig(Schema):
    """Base class for all resource types."""

    type: str = Field(init=False)
    """Type identifier for this resource."""

    description: str = ""
    """Human-readable description of the resource."""

    uri: str | None = None
    """Canonical URI for this resource, set during registration if unset."""

    # watch: WatchConfig | None = None
    # """Configuration for file system watching, if supported."""

    name: str | None = Field(None, exclude=True)
    """Technical identifier (automatically set from config key during registration)."""

    model_config = ConfigDict(frozen=True)

    # @property
    # def supports_watching(self) -> bool:
    #     """Whether this resource instance supports watching."""
    #     return False

    # def is_watched(self) -> bool:
    #     """Tell if this resource should be watched."""
    #     return self.supports_watching and self.watch is not None and self.watch.enabled

    def is_templated(self) -> bool:
        """Whether this resource supports URI templates."""
        return False  # Default: resources are static

    @property
    def mime_type(self) -> str:
        """Get the MIME type for this resource.

        This should be overridden by subclasses that can determine
        their MIME type. Default is text/plain.
        """
        return "text/plain"


class PathResourceLoaderConfig(BaseResourceLoaderConfig):
    """Resource loaded from a file or URL."""

    type: Literal["path"] = Field(default="path", init=False)
    """Discriminator field identifying this as a path-based resource."""

    path: str
    """Path to the file or URL to load."""

    watch: WatchConfig | None = None
    """Configuration for watching the file for changes."""

    def validate_resource(self) -> list[str]:
        """Check if path exists for local files."""
        import upath

        warnings = []
        path = upath.UPath(self.path)
        prefixes = ("http://", "https://")

        if not path.exists() and not path.as_uri().startswith(prefixes):
            warnings.append(f"Resource path not found: {path}")

        return warnings

    @property
    def supports_watching(self) -> bool:
        """Whether this resource instance supports watching."""
        import upath

        if not upath.UPath(self.path).exists():
            msg = f"Cannot watch non-existent path: {self.path}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True

    @model_validator(mode="after")
    def validate_path(self) -> PathResourceLoaderConfig:
        """Validate that the path is not empty."""
        if not self.path:
            msg = "Path cannot be empty"
            raise ValueError(msg)
        return self

    def is_templated(self) -> bool:
        """Path resources are templated if they contain placeholders."""
        return "{" in str(self.path)

    @property
    def mime_type(self) -> str:
        """Get MIME type based on file extension."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(self.path))
        return mime_type or "application/octet-stream"


class TextResourceLoaderConfig(BaseResourceLoaderConfig):
    """Raw text resource."""

    type: Literal["text"] = Field(default="text", init=False)
    """Discriminator field identifying this as a text-based resource."""

    content: str
    """The actual text content of the resource."""

    _mime_type: str | None = None  # Optional override

    @model_validator(mode="after")
    def validate_content(self) -> TextResourceLoaderConfig:
        """Validate that the content is not empty."""
        if not self.content:
            msg = "Content cannot be empty"
            raise ValueError(msg)
        return self

    @property
    def mime_type(self) -> str:
        """Get MIME type, trying to detect JSON/YAML."""
        if self._mime_type:
            return self._mime_type
        # Could add content inspection here
        return "text/plain"


class CLIResourceLoaderConfig(BaseResourceLoaderConfig):
    """Resource from CLI command execution."""

    type: Literal["cli"] = Field(default="cli", init=False)
    """Discriminator field identifying this as a CLI-based resource."""

    command: str | TypingSequence[str]
    """Command to execute (string or sequence of arguments)."""

    shell: bool = False
    """Whether to run the command through a shell."""

    cwd: str | None = None
    """Working directory for command execution."""

    timeout: float | None = None
    """Maximum time in seconds to wait for command completion."""

    @model_validator(mode="after")
    def validate_command(self) -> CLIResourceLoaderConfig:
        """Validate command configuration."""
        if not self.command:
            msg = "Command cannot be empty"
            raise ValueError(msg)
        if (
            isinstance(self.command, list | tuple)
            and not self.shell
            and not all(isinstance(part, str) for part in self.command)
        ):
            msg = "When shell=False, all command parts must be strings"
            raise ValueError(msg)
        return self


class RepositoryResource(BaseResourceLoaderConfig):
    """Git repository content."""

    type: Literal["repository"] = Field("repository", init=False)
    repo_url: str
    """URL of the git repository."""

    ref: str = "main"
    """Git reference (branch, tag, or commit)."""

    path: str = ""
    """Path within the repository."""

    sparse_checkout: list[str] | None = None
    """Optional list of paths for sparse checkout."""

    user: str | None = None
    """Optional user name for authentication."""

    password: SecretStr | None = None
    """Optional password for authentication."""

    def validate_resource(self) -> list[str]:
        warnings = []
        if self.user and not self.password:
            warnings.append(f"Repository {self.repo_url} has user but no password")
        return warnings


class SourceResourceLoaderConfig(BaseResourceLoaderConfig):
    """Resource from Python source code."""

    type: Literal["source"] = Field(default="source", init=False)
    """Discriminator field identifying this as a source code resource."""

    import_path: str
    """Dotted import path to the Python module or object."""

    recursive: bool = False
    """Whether to include submodules recursively."""

    include_tests: bool = False
    """Whether to include test files and directories."""

    @model_validator(mode="after")
    def validate_import_path(self) -> SourceResourceLoaderConfig:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self


class CallableResourceLoaderConfig(BaseResourceLoaderConfig):
    """Resource from executing a Python callable."""

    type: Literal["callable"] = Field(default="callable", init=False)
    """Discriminator field identifying this as a callable-based resource."""

    import_path: str
    """Dotted import path to the callable to execute."""

    keyword_args: JsonObject = Field(default_factory=dict)
    """Keyword arguments to pass to the callable."""

    @model_validator(mode="after")
    def validate_import_path(self) -> CallableResourceLoaderConfig:
        """Validate that the import path is properly formatted."""
        if not all(part.isidentifier() for part in self.import_path.split(".")):
            msg = f"Invalid import path: {self.import_path}"
            raise ValueError(msg)
        return self

    def is_templated(self) -> bool:
        """Callable resources are templated if they take parameters."""
        fn = import_callable(self.import_path)
        sig = inspect.signature(fn)
        return bool(sig.parameters)


class LangChainResourceLoader(BaseResourceLoaderConfig):
    """Wrapper for LangChain document loaders."""

    type: Literal["langchain"] = Field("langchain", init=False)

    loader_class: str
    """Import path to LangChain loader class."""

    loader_args: JsonObject = Field(default_factory=dict)
    """Arguments for loader initialization."""

    # async def load(self, **kwargs: Any) -> AsyncIterator[Content]:
    #     """Load documents using LangChain loader.

    #     Converts LangChain documents to Content objects.
    #     """
    #     from langchain.document_loaders import BaseLoader
    #     from llmling_agent.utils.importing import import_class

    #     # Import and initialize loader
    #     loader_cls = import_class(self.loader_class)
    #     if not issubclass(loader_cls, BaseLoader):
    #         msg = f"{self.loader_class} is not a LangChain loader"
    #         raise ValueError(msg)

    #     loader = loader_cls(**self.loader_args)

    #     # Load documents
    #     for doc in await loader.aload():
    #         yield Content(
    #             content=doc.page_content,
    #             metadata=Metadata(mime_type="text/plain", extra=doc.metadata),
    #         )


Resource = Annotated[
    PathResourceLoaderConfig
    | TextResourceLoaderConfig
    | CLIResourceLoaderConfig
    | SourceResourceLoaderConfig
    | LangChainResourceLoader
    | CallableResourceLoaderConfig,
    Field(discriminator="type"),
]


class WatchConfig(Schema):
    """Watch configuration for resources."""

    enabled: bool = False
    """Whether the watch is enabled"""

    patterns: list[str] | None = None
    """List of pathspec patterns (.gitignore style)"""

    ignore_file: str | None = None
    """Path to .gitignore-style file"""
