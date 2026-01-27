"""Config and provider routes."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import os
from typing import TYPE_CHECKING

from fastapi import APIRouter

from agentpool_server.opencode_server.converters import convert_toko_model_to_opencode
from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models import (
    Config,
    Mode,
    Model,
    ModelCost,
    ModelLimit,
    Provider,
    ProviderListResponse,
    ProvidersResponse,
)


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo


router = APIRouter(tags=["config"])

# Provider display names and environment variable mappings
PROVIDER_INFO: dict[str, tuple[str, list[str]]] = {
    "anthropic": ("Anthropic", ["ANTHROPIC_API_KEY"]),
    "openai": ("OpenAI", ["OPENAI_API_KEY"]),
    "google": ("Google", ["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
    "mistral": ("Mistral", ["MISTRAL_API_KEY"]),
    "groq": ("Groq", ["GROQ_API_KEY"]),
    "deepseek": ("DeepSeek", ["DEEPSEEK_API_KEY"]),
    "xai": ("xAI", ["XAI_API_KEY"]),
    "together": ("Together AI", ["TOGETHER_API_KEY"]),
    "perplexity": ("Perplexity", ["PERPLEXITY_API_KEY"]),
    "cohere": ("Cohere", ["COHERE_API_KEY"]),
    "fireworks": ("Fireworks AI", ["FIREWORKS_API_KEY"]),
    "openrouter": ("OpenRouter", ["OPENROUTER_API_KEY"]),
    "bedrock": ("AWS Bedrock", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
    "azure": ("Azure OpenAI", ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]),
    "vertex": ("Google Vertex AI", ["GOOGLE_APPLICATION_CREDENTIALS"]),
}


def _group_models_by_provider(
    models: list[TokoModelInfo],
) -> dict[str, list[TokoModelInfo]]:
    """Group models by their provider."""
    grouped: dict[str, list[TokoModelInfo]] = defaultdict(list)
    for model in models:
        # Skip embedding models - OpenCode is for chat/agent models
        if model.is_embedding:
            continue
        grouped[model.provider].append(model)
    return grouped


def _build_providers(models: list[TokoModelInfo]) -> list[Provider]:
    """Build Provider list from tokonomics models."""
    grouped = _group_models_by_provider(models)
    providers: list[Provider] = []

    for provider_id, provider_models in sorted(grouped.items()):
        # Get provider display info
        display_name, env_vars = PROVIDER_INFO.get(
            provider_id, (provider_id.title(), [f"{provider_id.upper()}_API_KEY"])
        )

        # Convert models to OpenCode format
        models_dict: dict[str, Model] = {}
        for toko_model in provider_models:
            opencode_model = convert_toko_model_to_opencode(toko_model)
            # Use id_override if available (e.g., "opus" for Claude Code SDK)
            model_key = toko_model.id_override or toko_model.id
            models_dict[model_key] = opencode_model

        provider = Provider(
            id=provider_id,
            name=display_name,
            env=env_vars,
            models=models_dict,
        )
        providers.append(provider)

    return providers


async def _get_available_models() -> list[TokoModelInfo]:
    """Fetch available models using tokonomics."""
    from tokonomics.model_discovery import get_all_models

    max_age = timedelta(days=7)  # Cache for a week
    return await get_all_models(max_age=max_age)


@router.get("/config")
async def get_config(state: StateDep) -> Config:
    """Get server configuration."""
    from agentpool_server.opencode_server.models.config import Keybinds, WatcherConfig

    # Initialize config if not yet set
    if state.config is None:
        state.config = Config()

    # Ensure keybinds are set with defaults
    if state.config.keybinds is None:
        state.config.keybinds = Keybinds()

    # Ensure watcher config is set with sensible defaults
    if state.config.watcher is None:
        state.config.watcher = WatcherConfig(
            ignore=["node_modules/**", "__pycache__/**", ".venv/**", "*.pyc", ".mypy_cache/**"]
        )

    # Set a default model if not already configured
    if state.config.model is None:
        try:
            # Get available models
            toko_models = await state.agent.get_available_models()
            if toko_models:
                providers = _build_providers(toko_models)

                # Find first connected provider and use its first model
                for provider in providers:
                    if any(os.environ.get(env) for env in provider.env) and provider.models:
                        first_model = next(iter(provider.models.keys()))
                        state.config.model = f"{provider.id}/{first_model}"
                        break
        except Exception:  # noqa: BLE001
            pass  # If we can't set a default, that's okay

    return state.config


@router.patch("/config")
async def update_config(state: StateDep, config_update: Config) -> Config:
    """Update server configuration.

    Only updates fields that are provided (non-None).
    Returns the complete updated config.
    """
    # Initialize config if not yet set
    if state.config is None:
        state.config = Config()

    # Update only the fields that were provided
    update_data = config_update.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        setattr(state.config, field_name, value)

    return state.config


async def _get_variants_from_agent(agent: object) -> dict[str, dict[str, object]]:
    """Get variants from agent's thought_level modes.

    Only supported for Codex and Claude Code agents which have static,
    known thought_level modes.

    Args:
        agent: The agent to get modes from

    Returns:
        Dict mapping variant names to empty config dicts (config is agent-internal)
    """
    from agentpool.agents.claude_code_agent import ClaudeCodeAgent
    from agentpool.agents.codex_agent import CodexAgent

    # Only Codex and Claude Code have static thought_level modes we can expose
    if not isinstance(agent, (CodexAgent, ClaudeCodeAgent)):
        return {}

    try:
        mode_categories = await agent.get_modes()
    except Exception:  # noqa: BLE001
        return {}
    for category in mode_categories:
        if category.id == "thought_level":
            # Convert modes to variants - the actual config is handled by set_mode
            return {mode.id: {} for mode in category.available_modes}
    return {}


def _apply_variants_to_providers(
    providers: list[Provider], variants: dict[str, dict[str, object]]
) -> list[Provider]:
    """Apply variants to all models in all providers.

    For agents with known thought_level modes (Codex, Claude Code),
    the same variants apply to all models.
    """
    if not variants:
        return providers

    updated_providers = []
    for provider in providers:
        updated_models = {
            model_id: model.model_copy(update={"variants": variants})
            for model_id, model in provider.models.items()
        }
        updated_providers.append(provider.model_copy(update={"models": updated_models}))
    return updated_providers


def _get_dummy_providers() -> list[Provider]:
    """Return a single dummy provider for testing."""
    dummy_model = Model(
        id="gpt-4o",
        name="GPT-4o",
        attachment=True,
        cost=ModelCost(input=5.0, output=15.0),
        limit=ModelLimit(context=128000.0, output=4096.0),
        reasoning=False,
        release_date="2024-05-13",
        temperature=True,
        tool_call=True,
    )
    dummy_provider = Provider(
        id="openai",
        name="OpenAI",
        env=["OPENAI_API_KEY"],
        models={"gpt-4o": dummy_model},
    )
    return [dummy_provider]


@router.get("/config/providers")
async def get_providers(state: StateDep) -> ProvidersResponse:
    """Get available providers and models from agent."""
    providers: list[Provider] = []
    # Try to get models from the agent
    try:
        toko_models = await state.agent.get_available_models()
        if toko_models:
            providers = _build_providers(toko_models)
    except Exception:  # noqa: BLE001
        pass  # Fall through to dummy providers

    # Fall back to dummy providers if no models available
    if not providers:
        providers = _get_dummy_providers()

    # Get variants from agent's thought_level modes (for Codex, Claude Code, etc.)
    variants = await _get_variants_from_agent(state.agent)
    if variants:
        providers = _apply_variants_to_providers(providers, variants)

    # Build default models map: use first model for each connected provider
    default_models: dict[str, str] = {}
    connected_providers = [
        provider.id for provider in providers if any(os.environ.get(env) for env in provider.env)
    ]

    for provider in providers:
        if provider.id in connected_providers and provider.models:
            # Simply use the first available model
            default_models[provider.id] = next(iter(provider.models.keys()))

    return ProvidersResponse(providers=providers, default=default_models)


@router.get("/provider")
async def list_providers(state: StateDep) -> ProviderListResponse:
    """List all providers."""
    import os

    providers: list[Provider] = []

    # Try to get models from the agent
    try:
        toko_models = await state.agent.get_available_models()
        if toko_models:
            providers = _build_providers(toko_models)
    except Exception:  # noqa: BLE001
        pass  # Fall through to dummy providers

    # Fall back to dummy providers if no models available
    if not providers:
        providers = _get_dummy_providers()

    # Get variants from agent's thought_level modes (for Codex, Claude Code, etc.)
    variants = await _get_variants_from_agent(state.agent)
    if variants:
        providers = _apply_variants_to_providers(providers, variants)

    # Determine which providers are "connected" based on env vars
    connected = [
        provider.id for provider in providers if any(os.environ.get(env) for env in provider.env)
    ]

    # Build default models map: use first model for each connected provider
    default_models: dict[str, str] = {}
    for provider in providers:
        if provider.id in connected and provider.models:
            # Simply use the first available model
            default_models[provider.id] = next(iter(provider.models.keys()))

    return ProviderListResponse(
        all=providers,
        default=default_models,
        connected=connected,
    )


@router.get("/mode")
async def list_modes(state: StateDep) -> list[Mode]:
    """List available modes."""
    _ = state  # unused for now
    return [
        Mode(
            name="default",
            tools={},
        )
    ]
