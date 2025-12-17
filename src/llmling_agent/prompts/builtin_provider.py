"""Builtin prompt provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.prompts.base import BasePromptProvider
from llmling_agent_config.system_prompts import StaticPromptConfig


if TYPE_CHECKING:
    from llmling_agent_config.system_prompts import StaticPromptConfig


class BuiltinPromptProvider(BasePromptProvider):
    """Default provider using system prompts."""

    supports_variables = True
    name = "builtin"

    def __init__(self, manifest_prompts: dict[str, StaticPromptConfig]) -> None:
        from jinjarope import Environment

        self.prompts = manifest_prompts
        self.env = Environment(autoescape=False, enable_async=True)

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get prompt content as string."""
        from jinja2 import Template, meta

        if name not in self.prompts:
            msg = f"Prompt not found: {name}"
            raise KeyError(msg)

        prompt = self.prompts[name]
        content = prompt.content

        # Parse template to find required variables
        ast = self.env.parse(content)
        required_vars = meta.find_undeclared_variables(ast)
        if variables and (unknown_vars := (set(variables) - required_vars)):
            vars_ = ", ".join(unknown_vars)
            msg = f"Unknown variables for prompt {name}: {vars_}"
            raise KeyError(msg)

        if required_vars:
            if not variables:
                req = ", ".join(required_vars)
                msg = f"Prompt {name} requires variables: {req}"
                raise KeyError(msg)

            # Check for missing required variables
            missing_vars = required_vars - set(variables or {})
            if missing_vars:
                vars_ = ", ".join(missing_vars)
                msg = f"Missing required variables for prompt {name}: {vars_}"
                raise KeyError(msg)

            try:
                template = Template(content, enable_async=True)
                content = await template.render_async(**variables)
            except Exception as e:
                msg = f"Failed to render prompt {name}: {e}"
                raise ValueError(msg) from e

        return content

    async def list_prompts(self) -> list[str]:
        """List available prompt identifiers."""
        return list(self.prompts.keys())
