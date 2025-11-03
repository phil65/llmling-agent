"""Prompt models for LLMling."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ImportString
from schemez import Schema
import upath

from llmling_agent.log import get_logger


logger = get_logger(__name__)

MessageContentType = Literal["text", "resource", "image_url", "image_base64"]
# Our internal role type (could include more roles)
MessageRole = Literal["system", "user", "assistant", "tool"]


class MessageContent(Schema):
    """Content item in a message."""

    type: MessageContentType
    content: str  # The actual content (text/uri/url/base64)
    alt_text: str | None = None  # For images or resource descriptions

    model_config = ConfigDict(frozen=True)


class PromptParameter(BaseModel):
    """Prompt argument with validation information."""

    name: str
    """Name of the argument as used in the prompt."""

    description: str | None = None
    """Human-readable description of the argument."""

    required: bool = False
    """Whether this argument must be provided when formatting the prompt."""

    type_hint: ImportString = Field(default="str")
    """Type annotation for the argument, defaults to str."""

    default: Any | None = None
    """Default value if argument is optional."""

    completion_function: ImportString | None = Field(default=None)
    """Optional function to provide argument completions."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class PromptMessage(BaseModel):
    """A message in a prompt template."""

    role: MessageRole
    content: str | MessageContent | list[MessageContent] = ""

    model_config = ConfigDict(frozen=True)


class BasePrompt(BaseModel):
    """Base class for all prompts."""

    name: str
    """Technical identifier (automatically set from config key during registration)."""

    title: str | None = None
    """Title of the prompt."""

    description: str
    """Human-readable description of what this prompt does."""

    arguments: list[PromptParameter] = Field(default_factory=list)
    """List of arguments that this prompt accepts."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional metadata for storing custom prompt information."""
    # messages: list[PromptMessage]

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class StaticPrompt(BasePrompt):
    """Static prompt defined by message list."""

    messages: list[PromptMessage]
    """List of messages that make up this prompt."""

    type: Literal["text"] = Field("text", init=False)
    """Discriminator field identifying this as a static text prompt."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class DynamicPrompt(BasePrompt):
    """Dynamic prompt loaded from callable."""

    import_path: str | Callable
    """Dotted import path to the callable that generates the prompt."""

    template: str | None = None
    """Optional template string for formatting the callable's output."""

    completions: dict[str, str] | None = None
    """Optional mapping of argument names to completion functions."""

    type: Literal["function"] = Field("function", init=False)
    """Discriminator field identifying this as a function-based prompt."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class FilePrompt(BasePrompt):
    """Prompt loaded from a file.

    This type of prompt loads its content from a file, allowing for longer or more
    complex prompts to be managed in separate files. The file content is loaded
    and parsed according to the specified format.
    """

    path: str | os.PathLike[str] | upath.UPath
    """Path to the file containing the prompt content."""

    fmt: Literal["text", "markdown", "jinja2"] = Field("text", alias="format")
    """Format of the file content (text, markdown, or jinja2 template)."""

    type: Literal["file"] = Field("file", init=False)
    """Discriminator field identifying this as a file-based prompt."""

    watch: bool = False
    """Whether to watch the file for changes and reload automatically."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


# Type to use in configuration
PromptType = Annotated[
    StaticPrompt | DynamicPrompt | FilePrompt, Field(discriminator="type")
]
