"""Config and provider routes."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

from fastapi import APIRouter

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

# Default model selection for each provider
# Maps provider ID to a model ID that should be used as default
# Based on latest models from models.dev as of 2026-01-13
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-5",  # Latest: 2025-11-24, with reasoning
    "openai": "gpt-5.2-pro",  # Latest: 2025-12-11, with reasoning
    "google": "gemini-3-flash-preview",  # Latest: 2025-12-17, with reasoning
    "mistral": "devstral-2512",  # Latest: 2025-12-09
    "groq": "moonshotai/kimi-k2-instruct-0905",  # Latest: 2025-09-05
    "deepseek": "deepseek-reasoner",  # Latest: 2025-01-20, with reasoning
    "xai": "grok-4-1-fast",  # Latest: 2025-11-19, with reasoning
    "perplexity": "sonar",  # Latest available
    "cohere": "command-a-translate-08-2025",  # Latest: 2025-08-28
    "openrouter": "minimax/minimax-m2.1",  # Latest: 2025-12-23, with reasoning
    "azure": "gpt-5.2-chat",  # Latest: 2025-12-11, with reasoning
    # Note: together, fireworks, bedrock, vertex not found in models.dev API
    # Keeping some fallbacks for compatibility
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "vertex": "gemini-2.0-flash-exp",
}


def _convert_toko_model_to_opencode(model: TokoModelInfo) -> Model:
    """Convert a tokonomics ModelInfo to an OpenCode Model."""
    # Convert pricing (tokonomics uses per-token, OpenCode uses per-million-token)
    input_cost = 0.0
    output_cost = 0.0
    cache_read = None
    cache_write = None

    if model.pricing:
        # tokonomics pricing is per-token, convert to per-million-tokens
        if model.pricing.prompt is not None:
            input_cost = model.pricing.prompt * 1_000_000
        if model.pricing.completion is not None:
            output_cost = model.pricing.completion * 1_000_000
        if model.pricing.input_cache_read is not None:
            cache_read = model.pricing.input_cache_read * 1_000_000
        if model.pricing.input_cache_write is not None:
            cache_write = model.pricing.input_cache_write * 1_000_000

    cost = ModelCost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read,
        cache_write=cache_write,
    )

    # Convert limits
    context = float(model.context_window) if model.context_window else 128000.0
    output = float(model.max_output_tokens) if model.max_output_tokens else 4096.0
    limit = ModelLimit(context=context, output=output)

    # Determine capabilities from modalities and metadata
    has_vision = "image" in model.input_modalities
    has_reasoning = "reasoning" in model.output_modalities or "thinking" in model.name.lower()

    # Format release date if available
    release_date = ""
    if model.created_at:
        release_date = model.created_at.strftime("%Y-%m-%d")

    return Model(
        id=model.id,
        name=model.name,
        attachment=has_vision,
        cost=cost,
        limit=limit,
        reasoning=has_reasoning,
        release_date=release_date,
        temperature=True,
        tool_call=True,  # Assume most models support tool calling
    )


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
            opencode_model = _convert_toko_model_to_opencode(toko_model)
            models_dict[toko_model.id] = opencode_model

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
    import os

    # Initialize config if not yet set
    if state.config is None:
        state.config = Config()

    # Set a default model if not already configured
    if state.config.model is None:
        try:
            # Get available models
            toko_models = await state.agent.get_available_models()
            if toko_models:
                providers = _build_providers(toko_models)

                # Find first connected provider
                for provider in providers:
                    if any(os.environ.get(env) for env in provider.env):
                        # Try to use predefined default
                        if provider.id in DEFAULT_MODELS:
                            default_model_id = DEFAULT_MODELS[provider.id]
                            if default_model_id in provider.models:
                                state.config.model = f"{provider.id}/{default_model_id}"
                                break
                        # Fall back to first available model
                        if provider.models:
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

    # Build default models map (same logic as /provider endpoint)
    default_models: dict[str, str] = {}
    connected_providers = [
        provider.id for provider in providers if any(os.environ.get(env) for env in provider.env)
    ]

    for provider in providers:
        # Only include connected providers
        if provider.id not in connected_providers:
            continue

        # Check if we have a predefined default for this provider
        if provider.id in DEFAULT_MODELS:
            default_model_id = DEFAULT_MODELS[provider.id]
            # Verify the model exists in the provider's model list
            if default_model_id in provider.models:
                default_models[provider.id] = default_model_id
            # If predefined default doesn't exist, pick first available model
            elif provider.models:
                default_models[provider.id] = next(iter(provider.models.keys()))
        # No predefined default, pick first available model
        elif provider.models:
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

    # Determine which providers are "connected" based on env vars
    connected = [
        provider.id for provider in providers if any(os.environ.get(env) for env in provider.env)
    ]

    # Build default models map for connected providers that have a known default
    default_models: dict[str, str] = {}
    for provider in providers:
        # Only include connected providers
        if provider.id not in connected:
            continue

        # Check if we have a predefined default for this provider
        if provider.id in DEFAULT_MODELS:
            default_model_id = DEFAULT_MODELS[provider.id]
            # Verify the model exists in the provider's model list
            if default_model_id in provider.models:
                default_models[provider.id] = default_model_id
            # If predefined default doesn't exist, pick first available model
            elif provider.models:
                default_models[provider.id] = next(iter(provider.models.keys()))
        # No predefined default, pick first available model
        elif provider.models:
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
