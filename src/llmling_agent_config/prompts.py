"""Prompt models for LLMling-Agent."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema
from upathtools import UPath

from llmling_agent.log import get_logger


logger = get_logger(__name__)

MessageContentType = Literal["text", "resource", "image_url", "image_base64"]
# Our internal role type (could include more roles)
MessageRole = Literal["system", "user", "assistant", "tool"]


class MessageContent(Schema):
    """Content item in a message."""

    type: MessageContentType = Field(
        title="Content type",
        examples=["text", "resource", "image_url"],
    )
    """Message content type."""

    content: str = Field(
        title="Content value",
        examples=["Hello world", "file:///path/to/file.txt", "data:image/png;base64,..."],
    )  # The actual content (text/uri/url/base64)
    """The actual content (text/uri/url/base64)."""

    alt_text: str | None = Field(
        default=None,
        title="Alternative text",
        examples=["Image description", "Resource summary"],
    )
    """Alternative text for the content."""

    model_config = ConfigDict(frozen=True)


class PromptParameter(Schema):
    """Prompt argument with validation information."""

    name: str = Field(
        title="Parameter name",
        examples=["user_name", "task_description", "max_tokens"],
    )
    """Name of the argument as used in the prompt."""

    description: str | None = Field(
        default=None,
        title="Parameter description",
        examples=["Name of the user", "Description of the task to perform"],
    )
    """Human-readable description of the argument."""

    required: bool = Field(default=False, title="Required parameter")
    """Whether this argument must be provided when formatting the prompt."""

    type_hint: ImportString[Any] = Field(
        default="str",
        title="Type hint",
        examples=["str", "int", "bool", "list[str]"],
    )
    """Type annotation for the argument, defaults to str."""

    default: Any | None = Field(default=None, title="Default value")
    """Default value if argument is optional."""


class PromptMessage(Schema):
    """A message in a prompt template."""

    role: MessageRole = Field(title="Message role", examples=["system", "user", "assistant"])
    """Role of the message."""

    content: str | MessageContent | list[MessageContent] = Field(
        default="",
        title="Message content",
        examples=["You are a helpful assistant", "What is the weather like?"],
    )
    """Content of the message."""


class BasePromptConfig(Schema):
    """Base class for all prompts."""

    name: str = Field(title="Prompt name", examples=["greeting", "code_review", "summarize_text"])
    """Technical identifier (automatically set from config key during registration)."""

    title: str | None = Field(
        default=None,
        title="Prompt title",
        examples=["Greeting Assistant", "Code Review Helper", "Text Summarizer"],
    )
    """Title of the prompt."""

    description: str = Field(
        title="Prompt description",
        examples=["A friendly greeting prompt", "Reviews code for best practices"],
    )
    """Human-readable description of what this prompt does."""

    arguments: list[PromptParameter] = Field(default_factory=list, title="Prompt arguments")
    """List of arguments that this prompt accepts."""

    metadata: dict[str, Any] = Field(default_factory=dict, title="Prompt metadata")
    """Additional metadata for storing custom prompt information."""
    # messages: list[PromptMessage]


class StaticPromptConfig(BasePromptConfig):
    """Static prompt defined by message list."""

    messages: list[PromptMessage] = Field(title="Prompt messages")
    """List of messages that make up this prompt."""

    type: Literal["text"] = Field(default="text", init=False, title="Prompt type")
    """Discriminator field identifying this as a static text prompt."""


class DynamicPromptConfig(BasePromptConfig):
    """Dynamic prompt loaded from callable."""

    import_path: str | Callable[..., Any] = Field(
        title="Import path",
        examples=["mymodule.create_prompt", "utils.prompts:generate_greeting"],
    )
    """Dotted import path to the callable that generates the prompt."""

    template: str | None = Field(
        default=None,
        title="Template string",
        examples=["Hello {{ name }}!", "Task: {{ task }}\nInstructions: {{ instructions }}"],
    )
    """Optional template string for formatting the callable's output."""

    completions: dict[str, str] | None = Field(default=None, title="Completion mappings")
    """Optional mapping of argument names to completion functions."""

    type: Literal["function"] = Field(default="function", init=False, title="Prompt type")
    """Discriminator field identifying this as a function-based prompt."""


class FilePromptConfig(BasePromptConfig):
    """Prompt loaded from a file.

    This type of prompt loads its content from a file, allowing for longer or more
    complex prompts to be managed in separate files. The file content is loaded
    and parsed according to the specified format.
    """

    path: str | os.PathLike[str] | UPath = Field(
        title="File path",
        examples=["prompts/greeting.txt", "/path/to/prompt.md", "templates/system.j2"],
    )
    """Path to the file containing the prompt content."""

    fmt: Literal["text", "markdown", "jinja2"] = Field(
        default="text",
        alias="format",
        title="File format",
        examples=["text", "markdown", "jinja2"],
    )
    """Format of the file content (text, markdown, or jinja2 template)."""

    type: Literal["file"] = Field(default="file", init=False, title="Prompt type")
    """Discriminator field identifying this as a file-based prompt."""

    watch: bool = Field(default=False, title="Watch for changes")
    """Whether to watch the file for changes and reload automatically."""


# Type to use in configuration
PromptConfig = Annotated[
    StaticPromptConfig | DynamicPromptConfig | FilePromptConfig,
    Field(discriminator="type"),
]
