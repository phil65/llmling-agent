"""System prompts configuration for agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema

from llmling_agent_config.prompt_hubs import PromptHubConfig


SystemPromptCategory = Literal["role", "methodology", "quality", "task"]

DEFAULT_TEMPLATE = """
{%- for prompt in role_prompts %}
Role: {{ prompt.content }}
{%- endfor %}

{%- for prompt in methodology_prompts %}
Method: {{ prompt.content }}
{%- endfor %}

{%- for prompt in quality_prompts %}
Quality Check: {{ prompt.content }}
{%- endfor %}

{%- for prompt in task_prompts %}
Task: {{ prompt.content }}
{%- endfor %}
"""


class BaseSystemPrompt(Schema):
    """Individual system prompt definition."""

    content: str
    """The actual prompt text."""

    category: SystemPromptCategory = "role"
    """Categorization for template organization."""

    model_config = ConfigDict(frozen=True)


class StaticPromptConfig(BaseSystemPrompt):
    """Configuration for a static text prompt."""

    type: Literal["static"] = Field("static", init=False)
    """Static prompt reference."""

    content: str
    """The prompt text content."""

    model_config = ConfigDict(frozen=True)


class FilePromptConfig(BaseSystemPrompt):
    """Configuration for a file-based Jinja template prompt."""

    type: Literal["file"] = Field("file", init=False)
    """File prompt reference."""

    path: str
    """Path to the Jinja template file."""

    variables: dict[str, Any] = Field(default_factory=dict)
    """Variables to pass to the template."""


class LibraryPromptConfig(BaseSystemPrompt):
    """Configuration for a library reference prompt."""

    type: Literal["library"] = Field("library", init=False)
    """Library prompt reference."""

    reference: str
    """Library prompt reference identifier."""


class FunctionPromptConfig(BaseSystemPrompt):
    """Configuration for a function-generated prompt."""

    type: Literal["function"] = Field("function", init=False)
    """Function prompt reference."""

    function: ImportString[Callable[..., str]]
    """Import path to the function that generates the prompt."""

    arguments: dict[str, Any] = Field(default_factory=dict)
    """Arguments to pass to the function."""


PromptConfig = Annotated[
    StaticPromptConfig | FilePromptConfig | LibraryPromptConfig | FunctionPromptConfig,
    Field(discriminator="type"),
]
"""Union type for different prompt configuration types."""


class PromptLibraryConfig(Schema):
    """Complete prompt configuration."""

    system_prompts: dict[str, StaticPromptConfig] = Field(default_factory=dict)
    """Mapping of system prompt identifiers to their definitions."""

    template: str | None = None
    """Optional template for combining prompts.
    Has access to prompts grouped by type."""

    providers: list[PromptHubConfig] = Field(default_factory=list)
    """List of external prompt providers to use."""

    model_config = ConfigDict(frozen=True)

    def format_prompts(self, identifiers: list[str] | None = None) -> str:
        """Format selected prompts using template.

        Args:
            identifiers: Optional list of prompt IDs to include.
                       If None, includes all prompts.
        """
        # Filter prompts if identifiers provided
        prompts = (
            {k: v for k, v in self.system_prompts.items() if k in identifiers}
            if identifiers
            else self.system_prompts
        )

        # Group prompts by type for template
        by_type = {
            "role_prompts": [p for p in prompts.values() if p.category == "role"],
            "methodology_prompts": [p for p in prompts.values() if p.category == "methodology"],
            "quality_prompts": [p for p in prompts.values() if p.category == "quality"],
            "task_prompts": [p for p in prompts.values() if p.category == "task"],
        }

        # Render template
        from jinja2 import Template

        template = Template(self.template or DEFAULT_TEMPLATE)
        return template.render(**by_type)
