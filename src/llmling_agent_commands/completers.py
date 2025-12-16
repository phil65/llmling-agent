"""Command completion providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

from slashed import CompletionItem, CompletionProvider

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.messaging.context import NodeContext  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from slashed import CompletionContext


def get_available_agents(
    ctx: CompletionContext[AgentContext[Any]],
    agent_type: Literal["all", "regular", "acp"] = "all",
) -> list[str]:
    """Get available agent names.

    Args:
        ctx: Completion context
        agent_type: Filter by agent type ("all", "regular", or "acp")
    """
    if not ctx.command_context.context.pool:
        return []
    pool = ctx.command_context.context.pool
    match agent_type:
        case "all":
            return list(pool.all_agents.keys())
        case "regular":
            return list(pool.agents.keys())
        case "acp":
            return list(pool.acp_agents.keys())


def get_available_nodes(ctx: CompletionContext[NodeContext[Any]]) -> list[str]:
    """Get available node names."""
    if not ctx.command_context.context.pool:
        return []
    return list(ctx.command_context.context.pool.nodes.keys())


def get_model_names(ctx: CompletionContext[AgentContext[Any]]) -> list[str]:
    """Get available model names from pydantic-ai and current configuration.

    Returns:
    - All models from KnownModelName literal type
    - Plus any additional models from current configuration
    """
    # Get models directly from the Literal type
    from llmling_models import AllModels
    from tokonomics.model_names import ModelName

    known_models = list(get_args(ModelName)) + list(get_args(AllModels))

    agent = ctx.command_context.context.agent
    if not agent.context.definition:
        return known_models

    # Add any additional models from the current configuration
    agents = agent.context.definition.agents
    config_models = {str(a.model) for a in agents.values() if a.model is not None}

    # Combine both sources, keeping order (known models first)
    all_models = known_models[:]
    all_models.extend(model for model in config_models if model not in all_models)
    return all_models


class PromptCompleter(CompletionProvider):
    """Completer for prompts."""

    async def get_completions(
        self, ctx: CompletionContext[AgentContext[Any]]
    ) -> AsyncIterator[CompletionItem]:
        """Complete prompt references."""
        current = ctx.current_word
        manifest = ctx.command_context.context.definition

        # If no : yet, suggest providers
        if ":" not in current:
            # Always suggest builtin prompts without prefix
            for name in manifest.prompts.system_prompts:
                if not name.startswith(current):
                    continue
                yield CompletionItem(name, metadata="Builtin prompt", kind="choice")

            # Suggest provider prefixes
            for provider in manifest.prompts.providers or []:
                prefix = f"{provider.type}:"
                if not prefix.startswith(current):
                    continue
                yield CompletionItem(prefix, metadata="Prompt provider", kind="choice")
            return

        # If after provider:, get prompts from that provider
        provider_, partial = current.split(":", 1)
        if provider_ == "builtin" or not provider_:
            # Complete from system prompts
            for name in manifest.prompts.system_prompts:
                if not name.startswith(partial):
                    continue
                text = f"{provider_}:{name}" if provider_ else name
                yield CompletionItem(text=text, metadata="Builtin prompt", kind="choice")
